# Late RWA effective Hamiltonian

![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22304140.svg)

This is the repository for the code to generate the data presented in [Effective Hamiltonian for an off-resonantly driven qubit-cavity system](https://doi.org/10.1103/y7xr-jq5w).

The main codebase is presented in the `src` folder, and is structured as follows:
- `hamiltonians.py` define the all the necessary Hamiltonians for the analysis.
- `config.py` define the system parameters used for the simulations presented in the paper.
- `stark_shifts.py` compute Stark shifts for the different models, and creates plots similar to Fig. 1 in the paper.
- `chevron.py` performs a two-dimensional parameter sweep of qubit and cavity drive amplitudes to recreate chevron patterns, such as those presented in Fig. 3-4 in the paper.
- `utils.py` define several utility and helper functions.

The chevron simulation is built on a Controller-Worker model defined in `mpi_utils.py`, allowing it to efficiently run on a distributed computing architecture, such as HPC. By setting the controller-worker class property `local` to true, the code can be ran on a local computer without MPI. Otherwise, the `mpi4py` package is required.

# License

This repository is licensed under the MIT License. See LICENSE file for details.
