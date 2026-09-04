import numpy as np

from iddefix.poleResidueFormulas import PoleResidue
from iddefix.resonatorFormulas import Impedances
from iddefix.poleResidueFitting import fit_residues


from scipy.integrate import quad


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


def test_finite_wake_impedance_matches_numerical_integration():
    frequency = 1.0e7
    wake_length = 20.0
    duration = wake_length / 299_792_458.0

    poles = np.array(
        [
            -2.0e7,
            -5.0e6 + 3.0e7j,
            -5.0e6 - 3.0e7j,
        ]
    )

    residues = np.array(
        [
            1.0e10,
            2.0e10 + 1.0e10j,
            2.0e10 - 1.0e10j,
        ]
    )

    def integrand(time):
        wake = np.sum(residues * np.exp(poles * time))
        return wake * np.exp(-2j * np.pi * frequency * time)

    real_part = quad(
        lambda time: integrand(time).real,
        0.0,
        duration,
    )[0]

    imaginary_part = quad(
        lambda time: integrand(time).imag,
        0.0,
        duration,
    )[0]

    expected = real_part + 1j * imaginary_part

    calculated = PoleResidue.finite_wake_impedance(
        frequencies=[frequency],
        poles=poles,
        residues=residues,
        wake_length=wake_length,
    )[0]

    np.testing.assert_allclose(
        calculated,
        expected,
        rtol=1.0e-10,
        atol=1.0e-8,
    )


def test_finite_wake_impedance_converges_to_full_impedance():
    frequencies = np.linspace(1.0e6, 1.0e8, 100)

    poles = np.array(
        [
            -2.0e7,
            -5.0e6 + 3.0e7j,
            -5.0e6 - 3.0e7j,
        ]
    )

    residues = np.array(
        [
            1.0e10,
            2.0e10 + 1.0e10j,
            2.0e10 - 1.0e10j,
        ]
    )

    finite_impedance = PoleResidue.finite_wake_impedance(
        frequencies,
        poles,
        residues,
        wake_length=5000.0,
    )

    full_impedance = PoleResidue.impedance(
        frequencies,
        poles,
        residues,
    )

    np.testing.assert_allclose(
        finite_impedance,
        full_impedance,
        rtol=1.0e-12,
        atol=1.0e-8,
    )


def test_zero_wake_length_produces_zero_impedance():
    impedance = PoleResidue.finite_wake_impedance(
        frequencies=[1.0e6, 2.0e6],
        poles=[-1.0e7],
        residues=[2.0e10],
        wake_length=0.0,
    )

    np.testing.assert_allclose(impedance, 0.0)


def test_least_squares_recovers_residues():
    frequencies = np.linspace(1.0e6, 2.0e8, 500)

    real_poles = np.array([-2.0e7])
    complex_poles = np.array([-5.0e6 + 4.0e7j])

    expected_poles = np.array(
        [
            real_poles[0],
            complex_poles[0],
            np.conj(complex_poles[0]),
        ]
    )

    expected_residues = np.array(
        [
            1.5e10,
            2.0e10 + 0.7e10j,
            2.0e10 - 0.7e10j,
        ]
    )

    impedance = PoleResidue.impedance(
        frequencies,
        expected_poles,
        expected_residues,
    )

    result = fit_residues(
        frequencies,
        impedance,
        real_poles,
        complex_poles,
    )

    np.testing.assert_allclose(
        result.fitted_impedance,
        impedance,
        rtol=1.0e-11,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        result.residues,
        expected_residues,
        rtol=1.0e-11,
        atol=1.0e-5,
    )


def test_least_squares_recovers_finite_wake_residues():
    frequencies = np.linspace(1.0e6, 2.0e8, 500)
    wake_length = 30.0

    real_poles = np.array([-2.0e7])
    complex_poles = np.array([-5.0e6 + 4.0e7j])

    expected_poles = np.array(
        [
            real_poles[0],
            complex_poles[0],
            np.conj(complex_poles[0]),
        ]
    )

    expected_residues = np.array(
        [
            1.5e10,
            2.0e10 + 0.7e10j,
            2.0e10 - 0.7e10j,
        ]
    )

    impedance = PoleResidue.finite_wake_impedance(
        frequencies,
        expected_poles,
        expected_residues,
        wake_length,
    )

    result = fit_residues(
        frequencies,
        impedance,
        real_poles,
        complex_poles,
        wake_length=wake_length,
    )

    np.testing.assert_allclose(
        result.fitted_impedance,
        impedance,
        rtol=1.0e-11,
        atol=1.0e-7,
    )

    np.testing.assert_allclose(
        result.residues,
        expected_residues,
        rtol=1.0e-11,
        atol=1.0e-5,
    )


def test_residue_fit_rejects_unstable_poles():
    with np.testing.assert_raises(ValueError):
        fit_residues(
            frequencies=[1.0e6],
            impedance=[1.0 + 1.0j],
            real_poles=[1.0e7],
            complex_poles=[],
        )


