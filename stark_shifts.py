from hamiltonians import (
    MultitoneDrives,
    early_rwa_hamiltonian,
    late_rwa_hamiltonian,
    get_snappa_betas,
)
import config
import utils
import numpy as np
import matplotlib.pyplot as plt



def calc_shifts(_, params):
    space = params["hilbert_space"]

    H = late_rwa_hamiltonian(**params)[0]
    energies, _ = utils.calculate_dressed_basis(
        *H.eigenstates(), space[0], space[1]
    )
    qubit_freq = energies[1] - energies[0]
    cavity_freq = energies[space[1]] - energies[0]

    H = late_rwa_hamiltonian(cr=False, **params)[0]
    energies, _ = utils.calculate_dressed_basis(
        *H.eigenstates(), space[0], space[1]
    )
    qubit_freq_no_cr = energies[1] - energies[0]
    cavity_freq_no_cr = energies[space[1]] - energies[0]
    return qubit_freq, cavity_freq, qubit_freq_no_cr, cavity_freq_no_cr


def calc_shifts_early_rwa(_, params):
    space = params["hilbert_space"]

    H = early_rwa_hamiltonian(**params)[0]
    energies, _ = utils.calculate_dressed_basis(
        *H.eigenstates(), space[0], space[1]
    )

    qubit_freq = energies[1] - energies[0]
    cavity_freq = energies[space[1]] - energies[0]

    return qubit_freq, cavity_freq


if __name__ == "__main__":

    parameters = config.parameters
    delta = 20 * 2 * np.pi
    qubit_beta, cavity_beta = get_snappa_betas()
    N = 41

    drives = MultitoneDrives()
    drives.add_qubit_drive(0.0, parameters["qubit_frequency"] - delta)
    drives.add_cavity_drive(
        0.0, parameters["cavity_frequency"] + delta - parameters["disp_shift"]
    )
    parameters["drives"] = drives
    space = parameters["hilbert_space"]

    qubit_amps = np.linspace(0, 0.08, N) * qubit_beta
    cavity_amps = np.linspace(0, 1.0, N) * cavity_beta

    results = utils.sweep(
        calc_shifts,
        parameters,
        {"drives.qubit_amplitudes.0": qubit_amps},
        progress_bar="tqdm",
        enumerate=True,
    )

    results_old = utils.sweep(
        calc_shifts_early_rwa,
        parameters,
        {"drives.qubit_amplitudes.0": qubit_amps},
        progress_bar="tqdm",
        enumerate=True,
    )

    qubit_freqs = np.empty((N,))
    qubit_freqs_no_cr = np.empty((N,))
    cavity_freqs_no_cr = np.empty((N,))
    qubit_freqs_old = np.empty((N,))
    cavity_freqs_old = np.empty((N,))
    cavity_freqs = np.empty((N,))
    for i, (qubit, cav, qubit_no_cr, cavity_no_cr) in enumerate(results):
        qubit_freqs[i] = qubit
        cavity_freqs[i] = cav
        qubit_freqs_no_cr[i] = qubit_no_cr
        cavity_freqs_no_cr[i] = cavity_no_cr

    for i, (qubit, cav) in enumerate(results_old):
        qubit_freqs_old[i] = qubit
        cavity_freqs_old[i] = cav

    qubit_freqs -= qubit_freqs[0]
    qubit_freqs_no_cr -= qubit_freqs_no_cr[0]
    qubit_freqs_old -= qubit_freqs_old[0]
    cavity_freqs -= cavity_freqs[0]
    cavity_freqs_no_cr -= cavity_freqs_no_cr[0]
    cavity_freqs_old -= cavity_freqs_old[0]

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    ax[0].plot(qubit_amps / (2 * np.pi), qubit_freqs / (2 * np.pi), label="Late RWA ($H_\\mathrm{eff}$)")
    ax[0].plot(qubit_amps / (2 * np.pi), qubit_freqs_no_cr / (2 * np.pi), "--", label="$H_\\mathrm{eff} - H_2$")
    ax[0].plot(qubit_amps / (2 * np.pi), qubit_freqs_old / (2 * np.pi), label="Early RWA")
    ax[0].grid(True)
    ax[0].set_axisbelow(True)
    ax[0].set_title("Stark shifts - Qubit drive")
    ax[0].set_ylabel("Stark shift [MHz]")
    ax[0].set_xlabel("Drive amplitude [MHz]")
    ax[0].legend()

    # parameters['drives'].qubit_amplitudes[0] = 0
    print(f"Cur. qubit amplitude {parameters['drives'].qubit_amplitudes[0]}") 
    cavity_amps = np.linspace(0, 1.00, N) * cavity_beta

    results = utils.sweep(
        calc_shifts,
        parameters,
        {"drives.cavity_amplitudes.0": cavity_amps},
        progress_bar="tqdm",
        enumerate=True,
    )

    results_old = utils.sweep(
        calc_shifts_early_rwa,
        parameters,
        {"drives.cavity_amplitudes.0": cavity_amps},
        progress_bar="tqdm",
        enumerate=True,
    )

    qubit_freqs = np.empty((N,))
    qubit_freqs_no_cr = np.empty((N,))
    qubit_freqs_old = np.empty((N,))
    cavity_freqs = np.empty((N,))
    cavity_freqs_old = np.empty((N,))
    cavity_freqs_no_cr = np.empty((N,))
    for i, (qubit, cav, qubit_no_cr, cavity_no_cr) in enumerate(results):
        qubit_freqs[i] = qubit
        cavity_freqs[i] = cav
        qubit_freqs_no_cr[i] = qubit_no_cr
        cavity_freqs_no_cr[i] = cavity_no_cr

    for i, (qubit, cav) in enumerate(results_old):
        qubit_freqs_old[i] = qubit
        cavity_freqs_old[i] = cav

    qubit_freqs -= qubit_freqs[0]
    cavity_freqs -= cavity_freqs[0]
    qubit_freqs_old -= qubit_freqs_old[0]
    qubit_freqs_no_cr -= qubit_freqs_no_cr[0]
    cavity_freqs_no_cr -= cavity_freqs_no_cr[0]
    cavity_freqs_old -= cavity_freqs_old[0]

    ax[1].plot(cavity_amps / (2 * np.pi), qubit_freqs / (2 * np.pi), label="Late RWA ($H_\\mathrm{eff}$)")
    ax[1].plot(cavity_amps / (2 * np.pi), qubit_freqs_no_cr / (2 * np.pi), "--", label="$H_\\mathrm{eff} - H_2$")
    ax[1].plot(cavity_amps / (2 * np.pi), qubit_freqs_old / (2 * np.pi), label="Early RWA")
    ax[1].grid(True)
    ax[1].set_axisbelow(True)
    ax[1].set_title("Stark shifts - Cavity drive")
    ax[1].set_ylabel("Stark shift [MHz]")
    ax[1].set_xlabel("Drive amplitude [MHz]")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
