import numpy as np

from iddefix.poleResidueFormulas import PoleResidue
from iddefix.resonatorFormulas import Impedances


def test_real_pole_wake():
    times = np.array([0.0, 1.0, 2.0])
    pole = -2.0
    residue = 3.0

    wake = PoleResidue.wake(times, [pole], [residue])
    expected = residue * np.exp(pole * times)

    np.testing.assert_allclose(wake, expected)


def test_complex_conjugate_pair_produces_real_wake():
    times = np.linspace(0.0, 2.0, 100)

    pole = -1.0 + 4.0j
    residue = 2.0 + 0.5j

    wake = PoleResidue.wake(
        times,
        poles=[pole, np.conj(pole)],
        residues=[residue, np.conj(residue)],
    )

    expected = 2.0 * np.real(
        residue * np.exp(pole * times)
    )

    assert np.isrealobj(wake)
    np.testing.assert_allclose(wake, expected)


def test_resonator_and_pole_residue_impedances_are_equal():
    frequencies = np.linspace(1.0e6, 2.0e9, 1000)

    Rs = 1.0e6
    Q = 5.0
    resonant_frequency = 8.0e8

    omega_r = 2.0 * np.pi * resonant_frequency

    poles = np.roots(
        [1.0, omega_r / Q, omega_r**2]
    )

    scale = Rs * omega_r / Q

    residues = np.array(
        [
            scale * poles[0] / (poles[0] - poles[1]),
            scale * poles[1] / (poles[1] - poles[0]),
        ]
    )

    impedance_resonator = Impedances.Resonator_longitudinal_imp(
        frequencies,
        Rs,
        Q,
        resonant_frequency,
    )

    impedance_poles = PoleResidue.impedance(
        frequencies,
        poles,
        residues,
    )

    np.testing.assert_allclose(
        impedance_poles,
        impedance_resonator,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


