""" This file contains code for simulating a two-dimensional drive amplitude sweep.

This code is able to reproduce Figure 3(a) in arXiv:2509.03375.

It utilizes MPI to access parallelize across an arbitrary number of nodes
using a controller-worker model.
"""
from hamiltonians import MultitoneDrives, late_rwa_hamiltonian, get_snappa_betas
from mpi_utils import MPIControllerWorker
import config
import utils

import numpy as np
import qutip as qt
import argparse

class ChevronSimulation(MPIControllerWorker):
    r"""
    Class for simulation a two-dimensional sweep of drive amplitude
    for the Late RWA Hamiltonian. The drive frequencies are set to
    drive a single tone transition |n,g> -> |n+1,e>.

    This utilizes MPIControllerWorker from `mpi_utils.py` to parallelize 
    across an arbitrary number of MPI nodes.

    The amplitudes to sweep are specified in the constructor, as well 
    as the target state n.
    """

    def __init__(self, qubit_amps, cavity_amps, target_state, gate_time=config.gate_time):
        super().__init__()

        self.qubit_amps = qubit_amps
        self.cavity_amps = cavity_amps
        self.target_state = target_state
        self.gatetime = gate_time

        self.params = config.parameters
        if target_state == 0 or target_state == 2:
            self.params["delta"] = 20 * 2 * np.pi
        else:
            self.params["delta"] = 30 * 2 * np.pi

        drives = MultitoneDrives()
        drives.add_qubit_drive(
            0.0, self.params["qubit_frequency"] - self.params["delta"]
        )
        drives.add_cavity_drive(
            0.0,
            self.params["cavity_frequency"]
            + self.params["delta"]
            - (target_state + 1) * self.params["disp_shift"],
        )
        self.params['drives'] = drives

        space = self.params['hilbert_space']

        self.initial_state = (qt.fock(space[0], target_state) & qt.fock(space[1], 0)).unit()
        self.tlist = np.linspace(0, self.gatetime, 1000)

    def worker(self, params, stage):
        ham, c_ops = late_rwa_hamiltonian(**params)

        options = {
            "nsteps": 100_000,
            "store_final_state": True,
            "store_states": False,
            "normalize_output": True
        }

        result = qt.mesolve(ham, self.initial_state, self.tlist, c_ops=c_ops, options=options)
        return np.real(result.final_state.ptrace(1).diag()[1])

    def controller(self):
        qubit_beta, cavity_beta = get_snappa_betas()

        print(f"Simulating chevron for target state {self.target_state} and gate time {self.gatetime}")

        param_list = utils.sweep(
            lambda x: (),
            self.params,
            {
                "drives.qubit_amplitudes.0": self.qubit_amps * qubit_beta,
                "drives.cavity_amplitudes.0": self.cavity_amps * cavity_beta,
            },
            return_params=True,
        )[1]

        X, Y = np.meshgrid(self.qubit_amps, self.cavity_amps)
        result = np.asarray(self.dispatch(param_list))
        result = result.reshape(X.shape)

        name = f"chevron_{self.target_state}_{self.gatetime}.npz"

        np.savez(name, qubit_amps = X, cavity_amps = Y, pops = result, target_state=target_state)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=int)
    parser.add_argument("--gatetime", type=float, default=config.gate_time)
    args = parser.parse_args()

    resolution = 31
    qubit_amps = np.linspace(0, 0.08, resolution)
    cavity_amps = np.linspace(0, 1, resolution)
    ChevronSimulation(qubit_amps, cavity_amps, args.state, gate_time=args.gatetime).run()
