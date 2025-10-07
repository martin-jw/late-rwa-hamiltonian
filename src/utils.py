from typing import Callable
import qutip as qt
import numpy as np
import copy


def _get_nested_param(obj, key):
    try:
        ind = int(key)
        return obj[ind]
    except (ValueError, IndexError):
        pass
    if isinstance(obj, dict):
        x = obj.get(key)
        if x:
            return x
    else: 
        x = getattr(obj, key, None)
        if x:
            return x

    raise ValueError(f"The object {obj} is missing key {key}")


def _set_nested_param(obj, key, value):
    try:
        ind = int(key)
        obj[ind] = value
        return
    except (ValueError, IndexError):
        pass
    if isinstance(obj, dict):
        if key in obj:
            obj[key] = value
            return
    if hasattr(obj, key):
        setattr(obj, key, value)
        return

    raise ValueError(f"The object {obj} is missing key {key}")


def _set_nested_params(params, kwargs):
    nested_keys = []
    for key in kwargs.keys():
        parts = key.split(".")
        if len(parts) > 1:
            obj = params
            for i in range(len(parts) - 1):
                part = parts[i]
                obj = _get_nested_param(obj, part)

            _set_nested_param(obj, parts[-1], kwargs[key])
            nested_keys.append(key)
            nested_keys.append(parts[0])
    return set(nested_keys)


def sweep(func: Callable, params: dict, sweep_params: dict, progress_bar: str = "", enumerate=False, return_params=False):

    params = copy.deepcopy(params)

    names = list(sweep_params.keys())
    values = list(sweep_params.values())

    grid_arrays = np.meshgrid(*values)
    for i in range(len(values)):
        values[i] = grid_arrays[i].flatten()

    num_iters = len(values[0])
    iter_range = range(num_iters)
    if progress_bar == 'tqdm':
        from tqdm import tqdm
        iter_range = tqdm(iter_range)

    results = [None] * num_iters
    plist = [None] * num_iters
    for j in iter_range:
        iter_params = {names[i]: values[i][j] for i in range(len(names))}
        nested = _set_nested_params(params, iter_params)
        non_nested = {
            key: iter_params[key] for key in iter_params.keys() if key not in nested
        }
        params.update(non_nested)
        plist[j] = copy.deepcopy(params)

        if enumerate:
            results[j] = func(j, params)
        else:
            results[j] = func(params)
    if return_params:
        return results, plist
    else:
        return results

def overlap(state1, state2):
    return np.abs(state1.overlap(state2)) ** 2

def calculate_dressed_basis(
    eigenvalues,
    eigenstates,
    cavity_levels,
    qubit_levels,
    return_sorting=False,
    force_sorting=None,
    start_zero=True,
    similarity_measure=overlap
):
    """Returns the eigenenergies and eigenstates for the dressed eigenbasis, sorted
    such that first eigenstate corresponds to |00>, second to |01>, third to |02>,
    fourth to |10>, etc. where the leftmost label is cavity. In the given
    example the qubit has three energy levels."""

    assert len(eigenstates) == qubit_levels * cavity_levels

    if force_sorting is not None:
        sorted_indices = force_sorting
        eigenenergies = np.array(eigenvalues)[sorted_indices]
        eigenenergies -= eigenenergies[0]
        if return_sorting:
            return (
                eigenenergies,
                [eigenstates[i] for i in sorted_indices],
                sorted_indices,
            )
        return eigenenergies, [eigenstates[i] for i in sorted_indices]

    # Overlaps between eigenstate i and bare index j
    overlaps = np.zeros((len(eigenstates), len(eigenstates)))
    for bare_index in range(len(eigenstates)):
        bare_state = qt.tensor(
            qt.fock(cavity_levels, bare_index // qubit_levels),
            qt.fock(qubit_levels, bare_index % qubit_levels),
        )
        for eig_index, eig in enumerate(eigenstates):
            overlaps[eig_index, bare_index] = similarity_measure(bare_state, eig)

    sorted_indices = -1 * np.ones(len(eigenstates), dtype=np.int32)

    remaining = list(range(len(eigenstates)))

    # Dressed eigenstates propose to the bare states
    while len(remaining) > 0:
        proposer = remaining.pop(0)
        olap = np.copy(overlaps[proposer, :])
        proposed = False
        while not proposed:
            proposee = np.argmax(olap)
            other = sorted_indices[proposee]
            if other >= 0:
                if overlaps[other, proposee] >= overlaps[proposer, proposee]:
                    olap[proposee] = 0
                else:
                    sorted_indices[proposee] = proposer
                    proposed = True
                    remaining.append(other)
            else:
                sorted_indices[proposee] = proposer
                proposed = True

    assert np.min(sorted_indices) >= 0
    assert len(np.unique(sorted_indices)) == len(eigenstates)

    eigenenergies = np.array(eigenvalues)[sorted_indices]
    if start_zero:
        eigenenergies -= eigenenergies[0]

    if return_sorting:
        return eigenenergies, [eigenstates[i] for i in sorted_indices], sorted_indices
    return eigenenergies, [eigenstates[i] for i in sorted_indices]

