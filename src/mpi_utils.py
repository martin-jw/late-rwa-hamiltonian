from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Sequence

from mpi4py import MPI
from qutip.solver.parallel import parallel_map


class ControlMessage(Enum):
    TERMINATE = 0
    DISPATCH = 1
    PROCEED = 2


class MPIControllerWorker(ABC):
    r"""Abstract class for MPI parallel processing using a Controller/Worker model.

    This class is an abstract class to simplify the processing of data in parallel
    using several MPI nodes. It uses a simplified Controller/Worker model, where the
    controller dispatches data to workers, which processes it. The controller then
    gathers the results of all workers.

    Since this is an abstract class it cannot be instanced directly. Instead, one
    should create a subclass which overrides the :py:meth:`MPIControllerWorker.worker`
    and :py:meth:`MPIControllerWorker.controller` methods.

    It uses the entire MPI world for execution. Each MPI node corresponds to one worker.
    The node with rank 0 is designated as controller. It should be noted that the
    controller is also a worker. That is, when the controller dispatches data to the
    workers for processing, the controller also receives data and processes it as
    a worker.

    Execution flow can be controlled through stages. Whenever the controller calls
    :py:meth:`MPIControllerWorker.dispatch`, the workers are called with a stage
    argument. This is an integer initialized to 0, and can be used to signal to the
    workers how to process the data. The controller can call :py:meth:`MPIControllerWorker.proceed`
    to increment the stage for all workers.

    Attributes
    ----------
    rank : int
        The MPI rank of the current process.
    world_size : int
        The total size of the MPI world, equal to the number of workers.
    name : int
        The name of the processor this process is running on.
    stage : int
        The current execution stage. See :py:meth:`MPIControllerWorker.worker`
        and :py:meth:`MPIControllerWorker.proceed` for details.

    Example
    -------
    The following is a simple example to highlight the basic usage of this class.

    ```python
    from mpi_utils import MPIControllerWorker

    class MPIExample(MPIControllerWorker):
        
        def __init__(self, local=False):
            # If local is True, no MPI is used, and the 
            # program parallelizes only on the local machine. 
            # This is useful for testing on machines without MPI installed.
            super().__init__(local=local)

            # Perform any initialization that needs to be shared by both
            # worker and controller here
            self.x_parameter = 5
            self.y_parameter = 2
    
        def worker(self, data, stage):
            # Override the worker abstract method,
            # this method is called on worker processes
            # when they receive data to process

            if stage == 0:
                # The first stage
                return data * self.x_parameter
            elif stage == 1:
                # The second stage
                return data / self.y_parameter
        
        def controller(self):
            # This method is only called on the controller process,
            # and is responsible for orchestrating the workers.
            # Any processing that needs to be on a single process
            # should happen here.

            initial_data = list(range(1000))
            
            # Dispatch the data to workers
            data_after_stage_0 = self.dispatch(initial_data)

            # Move to stage 1
            self.proceed()

            # Dispatch data to workers
            final_data = self.dispatch(data_after_stage_0)

            return final_data

    if __name__ == "__main__":
        output = MPIExample().run()
        print(output)
    ```
    """

    def __init__(self, local = False):
        self.local = local
        if local:
            self.rank = 0
            self.stage = 0
            self.world_size = 1
            self._comm = None
            self.name = "LOCAL"
        else:
            self._comm = MPI.COMM_WORLD
            self.rank = self._comm.Get_rank()
            self.world_size = self._comm.Get_size()
            self.name = MPI.Get_processor_name()
            self.stage = 0

    @abstractmethod
    def worker(self, data: Any, stage: int, *args, **kwargs) -> Any:
        r"""Called by worker processes to process data.

        This method is called by a worker process when it receives data
        to process from the controller. The execution stage can be used
        to identify how to process the data.

        Parameters
        ----------
        data : Any
            The data to process, received from the controller.
        stage : int
            The current execution stage, see also :py:meth:`MPIControllerWorker.proceed`.
        *args
            Other positional arguments received from dispatch.
        **kwargs
            Other keyword arguments received from dispatch.

        Returns
        -------
        Any
            The result of the data processing, depending on stage.
        """

        pass

    @abstractmethod
    def controller(self) -> Any:
        r"""Called by controller process to dispatch data for processing.

        This method is called by the controller process. It should create
        any data to be processed by the workers and dispatch it using
        :py:meth:`MPIControllerWorker.dispatch`. Use :py:meth:`MPIControllerWorker.proceed`
        to increment the execution stage.

        Returns
        -------
        Any
            The total result after processing all data. This can be `None` if
            all the necessary results are already outputted during the controller
            execution.
        """
        pass

    def _process_datalist(self, data_tuple: tuple):
        data_array, args, kwargs = data_tuple
        if self.local:
            result = parallel_map(self.worker, data_array, task_args=(self.stage, *args), task_kwargs=kwargs, progress_bar='tqdm')
        else:
            result = [None] * len(data_array)
            for i, data in enumerate(data_array):
                if data is not None:
                    result[i] = self.worker(data, self.stage, *args, **kwargs)
        return result

    def run(self, *args, **kwargs):
        r"""Start the controller and worker processes.

        Starts the execution by calling :py:meth:`MPIControllerWorker.controller`
        on the controller process and initializing the worker processes
        to start listening for data dispatched by the controller.

        NOTE: This method must be called on **ALL** processes, otherwise
        the program will block indefinitely.

        Parameters
        ----------
        *args
            Arguments to pass on to the controller method.
        **kwargs
            Keyword arguments to pass on to the controller method.

        Returns
        -------
        Any
            The return value from the controller method, after execution
            is finished.
        """

        self.stage = 0
        if self.local:
            return self._run_local(*args, **kwargs)

        if self.rank == 0:
            # This is the controller, run the controller method
            result = self.controller(*args, **kwargs)
            # Broadcast to workers that processing is over
            self._comm.bcast(ControlMessage.TERMINATE)
            return result

        else:
            # This is a worker, loop until ControlMessage.TERMINATE is received
            # from the controller.
            while True:
                # Wait for incoming control message
                control = self._comm.bcast(None, root=0)

                # Handle control message
                if control is ControlMessage.DISPATCH:
                    data_array = self._comm.scatter(None, root=0)
                    result = self._process_datalist(data_array)
                    self._comm.gather(result, root=0)
                elif control is ControlMessage.PROCEED:
                    self.stage += 1
                elif control is ControlMessage.TERMINATE:
                    break
                else:
                    raise RuntimeError("Worker received invalid control message!")

    def _run_local(self, *args, **kwargs):

        result = self.controller(*args, **kwargs)
        return result
        

    def proceed(self):
        r"""Increments the execution stage. Should only be called from controller.

        Raises
        ------
        AssertionError
            If called from a process which is not the controller process.
        """

        assert self.rank == 0, "proceed should only be called from the controller!"

        if not self.local:
            self._comm.bcast(ControlMessage.PROCEED, root=0)
        self.stage += 1
        print(f"Moving to stage {self.stage}")

    def _dispatch_local(self, data: Sequence[Any], *args, **kwargs):
        data_tuple = (data, args, kwargs)
        
        result = self._process_datalist(data_tuple)
        return result

    def dispatch(self, data: Sequence[Any], *args, **kwargs):
        r"""Dispatches data from the controller to the workers. Should only be called from controller.

        This method dispatches data specified by the `data` parameter as evenly
        as possible to all workers for processing. `data` should be an array, where
        each element is the individual data packets that will be passed to the
        :py:meth:`MPIControllerWorker.worker` method on all workers.

        This method blocks execution until all the data is processed, after which
        it returns the combined output of all worker processes. The result is
        guaranteed to be in the same order as `data`, that is `result[i]` is the
        output from the worker when processing `data[i]`.

        Parameters
        ----------
        data : Sequence[Any]
            The data to dispatch to the worker processes.
        *args
            Positional arguments to send as-is to all workers.
        **kwargs
            Keyword arguments to send as-is to all workers.

        Returns
        -------
        result : Sequence[Any]
            The result after `data` has been processed by the workers, sorted in the
            same order as `data`.

        Raises
        ------
        AssertionError
            If called from a process which is not the controller process.
        """

        assert self.rank == 0, "dispatch should only be called from the controller!"

        if self.local:
            return self._dispatch_local(data, *args, **kwargs)

        data_size = len(data)
        original_size = data_size
        if data_size % self.world_size != 0:
            new_size = (data_size // self.world_size + 1) * self.world_size
            data.extend([None] * (new_size - data_size))
            data_size = len(data)

        assert data_size % self.world_size == 0
        calls_per_node = data_size // self.world_size

        # Choose elements to dispatch to each node, such that
        # they are evenly sampled from the data array
        dispatch_list = [
            ([data[j + i * self.world_size] for i in range(calls_per_node)], args, kwargs)
            for j in range(self.world_size)
        ]

        # Notify workers that dispatch data is incoming
        self._comm.bcast(ControlMessage.DISPATCH, root=0)

        data_tuple = self._comm.scatter(dispatch_list, root=0)
        result = self._process_datalist(data_tuple)
        result = self._comm.gather(result, root=0)

        sorted_results = [None] * len(result) * calls_per_node
        for i in range(self.world_size):
            node_result = result[i]
            for j in range(calls_per_node):
                sorted_results[i + j * self.world_size] = node_result[j]

        if original_size < data_size:
            sorted_results = sorted_results[: (original_size - data_size)]
            data = data[: (original_size - data_size)]

        return sorted_results


__all__ = [MPIControllerWorker]
