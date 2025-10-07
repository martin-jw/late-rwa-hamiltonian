import qutip as qt
import numpy as np
from functools import lru_cache
import utils


def xi_amplitudes(
    qubit_frequency: float,
    cavity_frequency: float,
    qubit_drive_strength: float,
    qubit_drive_frequency: float,
    cavity_drive_strength: float,
    cavity_drive_frequency: float,
    **kwargs,
) -> tuple[float, float]:
    """_summary_

    Args:
        qubit_frequency (float): The qubit angular frequency, in MHz
        cavity_frequency (float): The cavity angular frequency, in MHz
        qubit_drive_strength (float): The qubit drive strength, in MHz
        qubit_drive_frequency (float): The qubit drive angular frequency, in MHz
        cavity_drive_strength (float): The cavity drive strength, in MHz
        cavity_drive_frequency (float): The cavity drive angular frequency, in MHz

    Returns:
        tuple[float, float]: The (xi1, xi2) for the time-independent effective
                                Hamiltonian of the SNAPPA gate.
    """

    xi1 = qubit_drive_strength / (2 * (qubit_frequency - qubit_drive_frequency))
    xi2 = cavity_drive_strength / (2 * (cavity_frequency - cavity_drive_frequency))

    return xi1, xi2


_qubit_drive_beta = 98.4145 * 2 * np.pi
_cavity_drive_beta = 45.0831 * 2 * np.pi


def get_snappa_betas():
    return _qubit_drive_beta, _cavity_drive_beta


class MultitoneDrives:
    def __init__(self, file=None):
        self.qubit_frequencies = []
        self.qubit_amplitudes = []
        self.qubit_phases = []
        self.cavity_frequencies = []
        self.cavity_amplitudes = []
        self.cavity_phases = []

        if file is not None:
            self.load(file)

    @property
    def num_qubit_drives(self):
        return len(self.qubit_frequencies)

    @property
    def num_cavity_drives(self):
        return len(self.cavity_frequencies)

    def get_drive_vector(
        self, include_amps=True, include_freqs=True, include_phases=True
    ):
        result = []
        if include_amps:
            result.extend(self.qubit_amplitudes)
        if include_freqs:
            result.extend(self.qubit_frequencies)
        if include_phases:
            result.extend(self.qubit_phases)
        if include_amps:
            result.extend(self.cavity_amplitudes)
        if include_freqs:
            result.extend(self.cavity_frequencies)
        if include_phases:
            result.extend(self.cavity_phases)
        return np.array(result)

    def from_drive_vector(
        self,
        vec,
        num_qubit_drives,
        num_cavity_drives,
        include_amps=True,
        include_freqs=True,
        include_phases=True,
    ):
        cur_ind = 0
        if include_amps:
            self.qubit_amplitudes = list(vec[:num_qubit_drives])
            cur_ind += num_qubit_drives
        if include_freqs:
            self.qubit_frequencies = list(vec[cur_ind : cur_ind + num_qubit_drives])
            cur_ind += num_qubit_drives
        if include_phases:
            self.qubit_phases = list(vec[cur_ind : cur_ind + num_qubit_drives])
            cur_ind += num_qubit_drives
        if include_amps:
            self.cavity_amplitudes = list(vec[cur_ind : cur_ind + num_cavity_drives])
            cur_ind += num_cavity_drives
        if include_freqs:
            self.cavity_frequencies = list(vec[cur_ind : cur_ind + num_cavity_drives])
            cur_ind += num_cavity_drives
        if include_phases:
            self.cavity_phases = list(vec[cur_ind:])

        if not include_phases:
            self.cavity_phases = [0] * num_cavity_drives
            self.qubit_phases = [0] * num_cavity_drives

    def add_qubit_drive(self, amplitude, frequency, phase=0.0):
        self.qubit_amplitudes.append(amplitude)
        self.qubit_frequencies.append(frequency)
        self.qubit_phases.append(phase)

    def add_cavity_drive(self, amplitude, frequency, phase=0.0):
        self.cavity_amplitudes.append(amplitude)
        self.cavity_frequencies.append(frequency)
        self.cavity_phases.append(phase)

    def iter_qubit(self):
        for i in range(self.num_qubit_drives):
            yield (
                self.qubit_amplitudes[i],
                self.qubit_frequencies[i],
                self.qubit_phases[i],
            )

    def iter_cavity(self):
        for i in range(self.num_cavity_drives):
            yield (
                self.cavity_amplitudes[i],
                self.cavity_frequencies[i],
                self.cavity_phases[i],
            )

    def qubit_xi(self, t, qubit_freq, omegaq_0, kappa=0):
        if self.num_qubit_drives == 0:
            return complex(0)

        def xi_term(amp, freq, phase):
            return complex(
                amp
                * np.exp(1j * phase)
                * np.exp(1j * (omegaq_0 - freq) * t)
                / (2 * (qubit_freq - freq) - 1j * kappa)
            )

        return np.sum(
            np.vectorize(xi_term)(
                self.qubit_amplitudes, self.qubit_frequencies, self.qubit_phases
            )
        )

    def qubit_xi_cr(self, t, qubit_freq, omegaq_0, kappa=0):
        if self.num_qubit_drives == 0:
            return complex(0)

        return complex(
            self.qubit_amplitudes[0]
            * np.exp(-1j * self.qubit_phases[0])
            / (2 * (qubit_freq + omegaq_0) - 1j * kappa)
        )

    def cavity_xi(self, t, cavity_freq, omegac_0, kappa=0):
        if self.num_cavity_drives == 0:
            return complex(0)

        def xi_term(amp, freq, phase):
            return complex(
                amp
                * np.exp(1j * phase)
                * np.exp(1j * (omegac_0 - freq) * t)
                / (2 * (cavity_freq - freq) - 1j * kappa)
            )

        return np.sum(
            np.vectorize(xi_term)(
                self.cavity_amplitudes, self.cavity_frequencies, self.cavity_phases
            )
        )

    def cavity_xi_cr(self, t, cavity_freq, omegac_0, kappa=0):
        if self.num_cavity_drives == 0:
            return complex(0)

        def xi_cr_term(amp, freq, phase):
            return complex(
                amp
                * np.exp(-1j * phase)
                * np.exp(1j * (freq - omegac_0) * t)
                / (2 * (cavity_freq + freq) - 1j * kappa)
            )

        return np.sum(
            np.vectorize(xi_cr_term)(
                self.cavity_amplitudes, self.cavity_frequencies, self.cavity_phases
            )
        )

    def save(self, path):
        np.savez(
            path,
            cav_amps=self.cavity_amplitudes,
            cav_freqs=self.cavity_frequencies,
            cav_phases=self.cavity_phases,
            qub_amps=self.qubit_amplitudes,
            qub_freqs=self.qubit_frequencies,
            qub_phases=self.qubit_phases,
        )

    def load(self, path):
        npzfile = np.load(path)
        if "x" in npzfile and "value" in npzfile:
            qubit_beta, cavity_beta = get_snappa_betas()
            import multitone_config as config
            x = npzfile["x"]
            self.qubit_amplitudes = [x[0] * qubit_beta]
            self.qubit_phases = [0.0]
            self.qubit_frequencies = [config.parameters['qubit_frequency'] - 30 * 2 * np.pi]
            num_states = (len(x) - 1)//3
            self.cavity_amplitudes = x[1: (1 + num_states)] * cavity_beta
            self.cavity_frequencies = x[(1 + num_states):(1 + 2 * num_states)]
            self.cavity_phases = x[(1 + 2 * num_states):(1 + 3 * num_states)]
        else:
            self.cavity_amplitudes = npzfile["cav_amps"]
            self.cavity_frequencies = npzfile["cav_freqs"]
            self.cavity_phases = npzfile["cav_phases"]
            self.qubit_amplitudes = npzfile["qub_amps"]
            self.qubit_frequencies = npzfile["qub_freqs"]
            self.qubit_phases = npzfile["qub_phases"]

    def __eq__(self, other):
        if not isinstance(other, MultitoneDrives):
            return NotImplemented
        return (
            self.qubit_amplitudes == other.qubit_amplitudes
            and self.qubit_frequencies == other.qubit_frequencies
            and self.qubit_phases == other.qubit_phases
            and self.cavity_amplitudes == other.cavity_amplitudes
            and self.cavity_frequencies == other.cavity_frequencies
            and self.cavity_phases == other.cavity_phases
        )


def late_rwa_hamiltonian(
    qubit_frequency,
    qubit_anharmonicity,
    cavity_frequency,
    cavity_kerr,
    disp_shift,
    disp_shift_corr,
    delta,
    drives: MultitoneDrives,
    hilbert_space,
    force_multitone=False,
    kappa_1q=0,
    kappa_2q=0,
    kappa_c=0,
    debug=False,
    **kwargs,
):
    """Returns the appropriate SNAPPA Hamiltonian based on the number of drives on each component."""

    # Select the most efficient Hamiltonian based on the number of drives
    if not force_multitone:
        single_qubit = False
        qubit_index = 0

        if drives.num_qubit_drives <= 1:
            single_qubit = True
            qubit_index = 0
        else:
            # See if only one qubit drive has non-zero amplitude
            num_non_zero = 0
            for i in range(drives.num_qubit_drives):
                if drives.qubit_amplitudes[i] > 0:
                    num_non_zero += 1
                    qubit_index = i

            if num_non_zero <= 1:
                single_qubit = True

        single_cavity = False
        cavity_index = 0
        if drives.num_cavity_drives <= 1:
            single_cavity = True
            cavity_index = 0
        else:
            # See if only one qubit drive has non-zero amplitude
            num_non_zero = 0
            for i in range(drives.num_cavity_drives):
                if drives.cavity_amplitudes[i] > 0:
                    num_non_zero += 1
                    cavity_index = i

            if num_non_zero <= 1:
                single_cavity = True

        if single_qubit:
            if single_cavity:
                if debug:
                    print("Choosing cavity_qubit_singletone_hamiltonian")
                return cavity_qubit_singletone_hamiltonian(
                    qubit_frequency,
                    qubit_anharmonicity,
                    cavity_frequency,
                    cavity_kerr,
                    disp_shift,
                    disp_shift_corr,
                    delta,
                    kappa_1q,
                    kappa_2q,
                    kappa_c,
                    drives,
                    hilbert_space,
                    qubit_drive_index=qubit_index,
                    cavity_drive_index=cavity_index,
                    **kwargs,
                )

            if debug:
                print("Choosing cavity_qubit_single_qubit_drive_hamiltonian")
            return cavity_qubit_single_qubit_drive_hamiltonian(
                qubit_frequency,
                qubit_anharmonicity,
                cavity_frequency,
                cavity_kerr,
                disp_shift,
                delta,
                kappa_1q,
                kappa_2q,
                kappa_c,
                drives,
                hilbert_space,
                **kwargs,
            )

    # No other Hamiltonian applies, return the full multitone Hamiltonian
    if debug:
        print("Choosing cavity_qubit_multitone_hamiltonian")
    return cavity_qubit_multitone_hamiltonian(
        qubit_frequency,
        qubit_anharmonicity,
        cavity_frequency,
        cavity_kerr,
        disp_shift,
        delta,
        kappa_1q,
        kappa_2q,
        kappa_c,
        drives,
        hilbert_space,
        **kwargs,
    )


def cavity_qubit_single_qubit_drive_hamiltonian(
    qubit_frequency,
    qubit_anharmonicity,
    cavity_frequency,
    cavity_kerr,
    disp_shift,
    delta,
    kappa_1q,
    kappa_2q,
    kappa_c,
    drives: MultitoneDrives,
    hilbert_space,
    **kwargs,
):
    """Cavity-qubit Hamiltonian driven by near-resonant drives, where the cavity drive is multitone,
    but the qubit drive is single tone."""
    cavity_levels, qubit_levels = hilbert_space
    b = qt.qeye(cavity_levels) & qt.destroy(qubit_levels)
    a = qt.destroy(cavity_levels) & qt.qeye(qubit_levels)

    # Ensure we only have a single qubit drive
    assert drives.num_qubit_drives <= 1, (
        "cavity_qubit_single_qubit_drive_hamiltonian requires at most ",
        "a single qubit drive, please use cavity_qubit_multitone_hamiltonian instead.",
    )

    # We use a fixed rotating frame for the cavity instead of basing it off the drive amplitudes.
    omegac_0 = cavity_frequency + delta - disp_shift

    # Ensure the qubit frame is rotating at the drive frequency
    if drives.num_qubit_drives == 1:
        omegaq_0 = drives.qubit_frequencies[0]
    else:
        omegaq_0 = qubit_frequency - delta

    # With a single qubit drive, xi1 is constant and real
    # as long as we're rotating at the drive frequency
    xi1 = drives.qubit_xi(0, qubit_frequency, omegaq_0, kappa=kappa_1q)
    xi1_cr = drives.qubit_xi_cr(0, qubit_frequency, omegaq_0, kappa=kappa_1q)
    if "cr" in kwargs and not kwargs["cr"]:
        xi1_cr = complex(0)

    @lru_cache
    def xi2(t):
        return drives.cavity_xi(t, cavity_frequency, omegac_0, kappa=kappa_c)

    @lru_cache
    def xi2_cr(t):
        if "cr" in kwargs and not kwargs["cr"]:
            return complex(0)
        return drives.cavity_xi_cr(t, cavity_frequency, omegac_0, kappa=kappa_c)

    static_diagonal = (
        -qubit_anharmonicity / 2 * b.dag() * b.dag() * b * b
        - cavity_kerr / 2 * a.dag() * a.dag() * a * a
        - disp_shift * b.dag() * b * a.dag() * a
    )

    static_offdiagonal = (
        -qubit_anharmonicity / 2 * xi1**2 * b.dag() * b.dag()
        + qubit_anharmonicity * (xi1 + xi1_cr.conjugate()) * b.dag() * b.dag() * b
        + disp_shift * (xi1 + xi1_cr.conjugate()) * b.dag() * a.dag() * a
    )
    static_offdiagonal += static_offdiagonal.dag()

    static_ham = static_diagonal + static_offdiagonal

    H = [
        static_ham,
        [
            b.dag() * b,
            lambda t, args: qubit_frequency
            - omegaq_0
            - 2 * qubit_anharmonicity * np.abs(xi1) ** 2
            - disp_shift * np.abs(xi2(t)) ** 2,
        ],
        [
            a.dag() * a,
            lambda t, args: cavity_frequency
            - omegac_0
            - 2 * cavity_kerr * np.abs(xi2(t)) ** 2
            - disp_shift * np.abs(xi1) ** 2,
        ],
        [-cavity_kerr / 2 * a.dag() * a.dag(), lambda t, args: xi2(t) ** 2],
        [-cavity_kerr / 2 * a * a, lambda t, args: xi2(t).conjugate() ** 2],
        [
            cavity_kerr * a.dag() * a.dag() * a,
            lambda t, args: xi2(t) + xi2_cr(t).conjugate(),
        ],
        [cavity_kerr * a.dag() * a * a, lambda t, args: xi2(t).conjugate() + xi2_cr(t)],
        [-disp_shift * xi1 * b.dag() * a.dag(), lambda t, args: xi2(t)],
        [-disp_shift * xi1 * b * a, lambda t, args: xi2(t).conjugate()],
        [
            -disp_shift * xi1_cr.conjugate() * b.dag() * a.dag(),
            lambda t, args: xi2_cr(t).conjugate(),
        ],
        [-disp_shift * xi1_cr * b * a, lambda t, args: xi2_cr(t)],
        [-disp_shift * xi1.conjugate() * b * a.dag(), lambda t, args: xi2(t)],
        [-disp_shift * xi1 * b.dag() * a, lambda t, args: xi2(t).conjugate()],
        [-disp_shift * xi1_cr * b * a.dag(), lambda t, args: xi2_cr(t).conjugate()],
        [-disp_shift * xi1_cr.conjugate() * b.dag() * a, lambda t, args: xi2_cr(t)],
        [
            disp_shift * b.dag() * b * a.dag(),
            lambda t, args: xi2(t) + xi2_cr(t).conjugate() / 6,
        ],
        [
            disp_shift * b.dag() * b * a,
            lambda t, args: xi2(t).conjugate() + xi2_cr(t) / 6,
        ],
        [
            b.dag(),
            lambda t, args: xi1
            * (
                qubit_anharmonicity * np.abs(xi1) ** 2
                + disp_shift * np.abs(xi2(t)) ** 2
            )
            + xi1_cr.conjugate() * (qubit_anharmonicity + disp_shift / 12),
        ],
        [
            b,
            lambda t, args: xi1
            * (
                qubit_anharmonicity * np.abs(xi1) ** 2
                + disp_shift * np.abs(xi2(t)) ** 2
            )
            + xi1_cr * (qubit_anharmonicity + disp_shift / 12),
        ],
        [
            a.dag(),
            lambda t, args: xi2(t)
            * (cavity_kerr * np.abs(xi2(t)) ** 2 + disp_shift * np.abs(xi1) ** 2)
            + xi2_cr(t).conjugate() * (cavity_kerr + disp_shift / 12),
        ],
        [
            a,
            lambda t, args: xi2(t).conjugate()
            * (cavity_kerr * np.abs(xi2(t)) ** 2 + disp_shift * np.abs(xi1) ** 2)
            + xi2_cr(t) * (cavity_kerr + disp_shift / 12),
        ],
    ]

    b_disp = b - xi1
    c_ops = [
        np.sqrt(kappa_c) * a,
        np.sqrt(kappa_1q) * b,
        np.sqrt(kappa_2q) * b_disp.dag() * b_disp,
    ]

    return qt.QobjEvo(H), c_ops


def cavity_qubit_multitone_hamiltonian(
    qubit_frequency,
    qubit_anharmonicity,
    cavity_frequency,
    cavity_kerr,
    disp_shift,
    delta,
    kappa_1q,
    kappa_2q,
    kappa_c,
    drives: MultitoneDrives,
    hilbert_space,
    **kwargs,
):
    cavity_levels, qubit_levels = hilbert_space
    b = qt.qeye(cavity_levels) & qt.destroy(qubit_levels)
    a = qt.destroy(cavity_levels) & qt.qeye(qubit_levels)

    # We use a fixed rotating frame instead of basing it off the drive amplitudes.
    omegac_0 = cavity_frequency + delta - disp_shift
    omegaq_0 = qubit_frequency - delta

    @lru_cache
    def xi1(t):
        res = drives.qubit_xi(t, qubit_frequency, omegaq_0, kappa=kappa_1q)
        return res

    @lru_cache
    def xi2(t):
        return drives.cavity_xi(t, cavity_frequency, omegac_0, kappa=kappa_c)

    static_ham = (
        -qubit_anharmonicity / 2 * b.dag() * b.dag() * b * b
        - cavity_kerr / 2 * a.dag() * a.dag() * a * a
        - disp_shift * b.dag() * b * a.dag() * a
    )

    H = [
        static_ham,
        [
            b.dag() * b,
            lambda t, args: qubit_frequency
            - omegaq_0
            - 2 * qubit_anharmonicity * np.abs(xi1(t)) ** 2
            - disp_shift * np.abs(xi2(t)) ** 2,
        ],
        [
            a.dag() * a,
            lambda t, args: cavity_frequency
            - omegac_0
            - 2 * cavity_kerr * np.abs(xi2(t)) ** 2
            - disp_shift * np.abs(xi1(t)) ** 2,
        ],
        [-qubit_anharmonicity / 2 * b.dag() * b.dag(), lambda t, args: xi1(t) ** 2],
        [-qubit_anharmonicity / 2 * b * b, lambda t, args: xi1(t).conjugate() ** 2],
        [qubit_anharmonicity * b.dag() * b.dag() * b, lambda t, args: xi1(t)],
        [qubit_anharmonicity * b.dag() * b * b, lambda t, args: xi1(t).conjugate()],
        [-cavity_kerr / 2 * a.dag() * a.dag(), lambda t, args: xi2(t) ** 2],
        [-cavity_kerr / 2 * a * a, lambda t, args: xi2(t).conjugate() ** 2],
        [cavity_kerr * a.dag() * a.dag() * a, lambda t, args: xi2(t)],
        [cavity_kerr * a.dag() * a * a, lambda t, args: xi2(t).conjugate()],
        [-disp_shift * b.dag() * a.dag(), lambda t, args: xi1(t) * xi2(t)],
        [-disp_shift * b * a, lambda t, args: xi1(t).conjugate() * xi2(t).conjugate()],
        [-disp_shift * b * a.dag(), lambda t, args: xi1(t).conjugate() * xi2(t)],
        [-disp_shift * b.dag() * a, lambda t, args: xi1(t) * xi2(t).conjugate()],
        [disp_shift * b.dag() * b * a.dag(), lambda t, args: xi2(t)],
        [disp_shift * b.dag() * b * a, lambda t, args: xi2(t).conjugate()],
        [disp_shift * b.dag() * a.dag() * a, lambda t, args: xi1(t)],
        [disp_shift * b * a.dag() * a, lambda t, args: xi1(t).conjugate()],
        [
            b.dag(),
            lambda t, args: xi1(t)
            * (
                qubit_anharmonicity * np.abs(xi1(t)) ** 2
                + disp_shift * np.abs(xi2(t)) ** 2
            ),
        ],
        [
            b,
            lambda t, args: xi1(t).conjugate()
            * (
                qubit_anharmonicity * np.abs(xi1(t)) ** 2
                + disp_shift * np.abs(xi2(t)) ** 2
            ),
        ],
        [
            a.dag(),
            lambda t, args: xi2(t)
            * (cavity_kerr * np.abs(xi2(t)) ** 2 + disp_shift * np.abs(xi1(t)) ** 2),
        ],
        [
            a,
            lambda t, args: xi2(t).conjugate()
            * (cavity_kerr * np.abs(xi2(t)) ** 2 + disp_shift * np.abs(xi1(t)) ** 2),
        ],
    ]

    c_ops = [
        np.sqrt(kappa_c) * a,
        np.sqrt(kappa_1q) * b,
        np.sqrt(kappa_2q) * b.dag() * b,
    ]
    return qt.QobjEvo(H), c_ops


def early_rwa_hamiltonian(
    qubit_frequency,
    qubit_anharmonicity,
    cavity_frequency,
    cavity_kerr,
    disp_shift,
    disp_shift_corr,
    delta,
    kappa_1q,
    kappa_2q,
    kappa_c,
    drives: MultitoneDrives,
    hilbert_space,
    qubit_drive_index=None,
    cavity_drive_index=None,
    **kwargs,
) -> qt.Qobj:
    cavity_levels, qubit_levels = hilbert_space
    b = qt.qeye(cavity_levels) & qt.destroy(qubit_levels)
    a = qt.destroy(cavity_levels) & qt.qeye(qubit_levels)

    if qubit_drive_index is None:
        qubit_drive_index = 0
    if cavity_drive_index is None:
        cavity_drive_index = 0

    if drives.num_qubit_drives > 0:
        qubit_drive_frequency = drives.qubit_frequencies[qubit_drive_index]
    else:
        qubit_drive_frequency = qubit_frequency

    if drives.num_cavity_drives > 0:
        cavity_drive_frequency = drives.cavity_frequencies[cavity_drive_index]
    else:
        cavity_drive_frequency = cavity_frequency

    xi1 = drives.qubit_xi(0, qubit_frequency, qubit_drive_frequency, kappa=kappa_1q)
    xi1_cr = drives.qubit_xi_cr(0, qubit_frequency, qubit_drive_frequency, kappa=kappa_1q)
    xi2 = drives.cavity_xi(0, cavity_frequency, cavity_drive_frequency, kappa=kappa_c)
    xi2_cr = drives.cavity_xi_cr(0, cavity_frequency, cavity_drive_frequency, kappa=kappa_c)

    q_freq = (
        qubit_frequency
        - qubit_drive_frequency
        - 2 * qubit_anharmonicity * (xi1.conjugate() * xi1)
        - disp_shift * (xi2.conjugate() * xi2)
    )
    c_freq = (
        cavity_frequency
        - cavity_drive_frequency
        - 2 * cavity_kerr * (xi2.conjugate() * xi2)
        - disp_shift * (xi1.conjugate() * xi1)
    )

    H0 = (
        q_freq * b.dag() * b
        + c_freq * a.dag() * a
        - qubit_anharmonicity / 2 * b.dag() * b.dag() * b * b
        - cavity_kerr / 2 * a.dag() * a.dag() * a * a
        - disp_shift * b.dag() * b * a.dag() * a
        - disp_shift_corr / 2 * a.dag() * a.dag() * a * a * b.dag() * b
    )

    H_coupling = 0
    H_coupling += (
        -disp_shift
        * (xi1 * xi2 + xi1_cr.conjugate() * xi2_cr.conjugate() / 6)
        * b.dag()
        * a.dag()
    )
    H_coupling += (
        -disp_shift
        * (xi1.conjugate() * xi2 + xi1_cr * xi2_cr.conjugate() / 6)
        * b
        * a.dag()
    )
    H_coupling += H_coupling.dag()

    H = H0 + H_coupling

    b_disp = b - xi1
    c_ops = [
        np.sqrt(kappa_c) * a,
        np.sqrt(kappa_1q) * b,
        np.sqrt(kappa_2q) * b_disp.dag() * b_disp,
    ]

    return H, c_ops


def cavity_qubit_singletone_hamiltonian(
    qubit_frequency,
    qubit_anharmonicity,
    cavity_frequency,
    cavity_kerr,
    disp_shift,
    disp_shift_corr,
    delta,
    kappa_1q,
    kappa_2q,
    kappa_c,
    drives: MultitoneDrives,
    hilbert_space,
    qubit_drive_index=None,
    cavity_drive_index=None,
    **kwargs,
) -> qt.Qobj:
    cavity_levels, qubit_levels = hilbert_space
    b = qt.qeye(cavity_levels) & qt.destroy(qubit_levels)
    a = qt.destroy(cavity_levels) & qt.qeye(qubit_levels)

    if qubit_drive_index is None:
        qubit_drive_index = 0
    if cavity_drive_index is None:
        cavity_drive_index = 0

    if drives.num_qubit_drives > 0:
        qubit_drive_frequency = drives.qubit_frequencies[qubit_drive_index]
    else:
        qubit_drive_frequency = qubit_frequency

    if drives.num_cavity_drives > 0:
        cavity_drive_frequency = drives.cavity_frequencies[cavity_drive_index]
    else:
        cavity_drive_frequency = cavity_frequency

    xi1 = drives.qubit_xi(0, qubit_frequency, qubit_drive_frequency, kappa=kappa_1q)
    xi1_cr = drives.qubit_xi_cr(
        0, qubit_frequency, qubit_drive_frequency, kappa=kappa_1q
    )
    if "cr" in kwargs and not kwargs["cr"]:
        xi1_cr = complex(0)
    xi2 = drives.cavity_xi(0, cavity_frequency, cavity_drive_frequency, kappa=kappa_c)
    xi2_cr = drives.cavity_xi_cr(
        0, cavity_frequency, cavity_drive_frequency, kappa=kappa_c
    )
    if "cr" in kwargs and not kwargs["cr"]:
        xi2_cr = complex(0)

    q_freq = (
        qubit_frequency
        - qubit_drive_frequency
        - 2 * qubit_anharmonicity * (xi1.conjugate() * xi1)
        - disp_shift * (xi2.conjugate() * xi2)
    )
    c_freq = (
        cavity_frequency
        - cavity_drive_frequency
        - 2 * cavity_kerr * (xi2.conjugate() * xi2)
        - disp_shift * (xi1.conjugate() * xi1)
    )

    H0 = (
        q_freq * b.dag() * b
        + c_freq * a.dag() * a
        - qubit_anharmonicity / 2 * b.dag() * b.dag() * b * b
        - cavity_kerr / 2 * a.dag() * a.dag() * a * a
        - disp_shift * b.dag() * b * a.dag() * a
        - disp_shift_corr / 2 * a.dag() * a.dag() * a * a * b.dag() * b
    )

    H_alpha = 0
    H_alpha += -qubit_anharmonicity / 2 * (xi1**2) * b.dag() * b.dag()
    H_alpha += qubit_anharmonicity * (xi1 + xi1_cr.conjugate()) * b.dag() * b.dag() * b
    H_alpha += H_alpha.dag()

    H_kerr = 0
    H_kerr += -cavity_kerr / 2 * (xi2**2) * a.dag() * a.dag()
    H_kerr += cavity_kerr * (xi2 + xi2_cr.conjugate()) * a.dag() * a.dag() * a
    H_kerr += H_kerr.dag()

    H_coupling = 0
    H_coupling += (
        -disp_shift
        * (xi1 * xi2 + xi1_cr.conjugate() * xi2_cr.conjugate() / 6)
        * b.dag()
        * a.dag()
    )
    H_coupling += (
        -disp_shift
        * (xi1.conjugate() * xi2 + xi1_cr * xi2_cr.conjugate() / 6)
        * b
        * a.dag()
    )
    H_coupling += disp_shift * (xi2 + xi2_cr.conjugate() / 6) * b.dag() * b * a.dag()
    H_coupling += disp_shift * (xi1 + xi1_cr.conjugate() / 6) * b.dag() * a.dag() * a
    H_coupling += H_coupling.dag()

    H_linear = 0
    H_linear += qubit_anharmonicity * xi1 * (xi1.conjugate() * xi1) * b.dag()
    H_linear += cavity_kerr * xi2 * (xi2.conjugate() * xi2) * a.dag()
    H_linear += disp_shift * xi1 * (xi2.conjugate() * xi2) * b.dag()
    H_linear += disp_shift * xi2 * (xi1.conjugate() * xi1) * a.dag()
    H_linear += (qubit_anharmonicity + disp_shift / 12) * xi1_cr.conjugate() * b.dag()
    H_linear += (cavity_kerr + disp_shift / 12) * xi2_cr.conjugate() * a.dag()
    H_linear += H_linear.dag()

    H = H0 + H_alpha + H_kerr + H_coupling + H_linear

    b_disp = b - xi1
    c_ops = [
        np.sqrt(kappa_c) * a,
        np.sqrt(kappa_1q) * b,
        np.sqrt(kappa_2q) * b_disp.dag() * b_disp,
    ]

    return H, c_ops

