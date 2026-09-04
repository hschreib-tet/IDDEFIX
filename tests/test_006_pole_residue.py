import numpy as np

from iddefix.poleResidueFormulas import PoleResidue
from iddefix.resonatorFormulas import Impedances
from iddefix.poleResidueFitting import (
    decode_log_poles,
    fit_poles_evolutionary,
    fit_residues,
    pole_objective,
)


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

def test_decode_log_poles():
    parameters = np.log10(
        [
            2.0e7,
            8.0e7,
            5.0e6,
            1.0e7,
            4.0e7,
            9.0e7,
        ]
    )

    real_poles, complex_poles = decode_log_poles(
        parameters,
        number_real_poles=2,
        number_complex_pairs=2,
    )

    expected_real_poles = np.array(
        [-2.0e7, -8.0e7]
    )

    expected_complex_poles = np.array(
        [
            -5.0e6 + 4.0e7j,
            -1.0e7 + 9.0e7j,
        ]
    )

    np.testing.assert_allclose(
        real_poles,
        expected_real_poles,
    )

    np.testing.assert_allclose(
        complex_poles,
        expected_complex_poles,
    )


def test_pole_objective_is_zero_for_exact_poles():
    frequencies = np.linspace(1.0e6, 2.0e8, 500)

    real_pole = -2.0e7
    complex_pole = -5.0e6 + 4.0e7j

    poles = np.array(
        [
            real_pole,
            complex_pole,
            np.conj(complex_pole),
        ]
    )

    residues = np.array(
        [
            1.5e10,
            2.0e10 + 0.7e10j,
            2.0e10 - 0.7e10j,
        ]
    )

    impedance = PoleResidue.impedance(
        frequencies,
        poles,
        residues,
    )

    parameters = np.log10(
        [
            2.0e7,
            5.0e6,
            4.0e7,
        ]
    )

    error = pole_objective(
        parameters,
        frequencies,
        impedance,
        number_real_poles=1,
        number_complex_pairs=1,
    )

    assert error < 1.0e-20


def test_exact_poles_fit_better_than_incorrect_poles():
    frequencies = np.linspace(1.0e6, 2.0e8, 500)

    real_pole = -2.0e7
    complex_pole = -5.0e6 + 4.0e7j

    poles = np.array(
        [
            real_pole,
            complex_pole,
            np.conj(complex_pole),
        ]
    )

    residues = np.array(
        [
            1.5e10,
            2.0e10 + 0.7e10j,
            2.0e10 - 0.7e10j,
        ]
    )

    impedance = PoleResidue.impedance(
        frequencies,
        poles,
        residues,
    )

    exact_parameters = np.log10(
        [2.0e7, 5.0e6, 4.0e7]
    )

    incorrect_parameters = np.log10(
        [8.0e7, 3.0e7, 1.2e8]
    )

    exact_error = pole_objective(
        exact_parameters,
        frequencies,
        impedance,
        number_real_poles=1,
        number_complex_pairs=1,
    )

    incorrect_error = pole_objective(
        incorrect_parameters,
        frequencies,
        impedance,
        number_real_poles=1,
        number_complex_pairs=1,
    )

    assert exact_error < incorrect_error


def test_evolutionary_fit_recovers_single_real_pole():
    frequencies = np.linspace(1.0e6, 1.0e8, 200)

    expected_pole = -2.0e7
    expected_residue = 1.5e10

    impedance = PoleResidue.impedance(
        frequencies,
        poles=[expected_pole],
        residues=[expected_residue],
    )

    result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=impedance,
        number_real_poles=1,
        number_complex_pairs=0,
        parameter_bounds=[
            (
                np.log10(1.0e7),
                np.log10(4.0e7),
            )
        ],
        maxiter=100,
        popsize=10,
        tol=1.0e-10,
        polish=True,
        seed=1234,
    )

    np.testing.assert_allclose(
        result.real_poles,
        [expected_pole],
        rtol=1.0e-5,
    )

    np.testing.assert_allclose(
        result.residue_fit.residues,
        [expected_residue],
        rtol=1.0e-5,
    )

    assert result.objective_value < 1.0e-12


def test_evolutionary_fit_recovers_complex_pole_pair():
    frequencies = np.linspace(1.0e6, 3.0e7, 300)

    expected_pole = -5e6 + 8.0e7j
    expected_residue = 2.0e10 + 0.5e10j

    impedance = PoleResidue.impedance(
        frequencies,
        poles=[
            expected_pole,
            np.conj(expected_pole),
        ],
        residues=[
            expected_residue,
            np.conj(expected_residue),
        ],
    )

    result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=impedance,
        number_real_poles=0,
        number_complex_pairs=1,
        parameter_bounds=[
            (
                np.log10(1.0e6),
                np.log10(1.0e7),
            ),
            (
                np.log10(5.0e7),
                np.log10(1.2e8),
            ),
        ],
        maxiter=150,
        popsize=12,
        tol=1.0e-9,
        polish=True,
        seed=1234,
        workers=1,
    )

    fitted_pole = result.complex_poles[0]
    fitted_residue = result.residue_fit.residues[0]

    np.testing.assert_allclose(
        fitted_pole,
        expected_pole,
        rtol=1.0e-4,
    )

    np.testing.assert_allclose(
        fitted_residue,
        expected_residue,
        rtol=1.0e-4,
    )

    assert result.objective_value < 1.0e-10

def test_evolutionary_fit_recovers_mixed_poles():
    frequencies = np.linspace(1.0e6, 3.0e7, 400)

    expected_real_pole = -1.0e7
    expected_complex_pole = -3.0e6 + 8.0e7j

    expected_real_residue = 8.0e9
    expected_complex_residue = 2.0e10 + 0.5e10j

    poles = np.array(
        [
            expected_real_pole,
            expected_complex_pole,
            np.conj(expected_complex_pole),
        ]
    )

    residues = np.array(
        [
            expected_real_residue,
            expected_complex_residue,
            np.conj(expected_complex_residue),
        ]
    )

    impedance = PoleResidue.impedance(
        frequencies,
        poles,
        residues,
    )

    result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=impedance,
        number_real_poles=1,
        number_complex_pairs=1,
        parameter_bounds=[
            # Real decay rate
            (
                np.log10(5.0e6),
                np.log10(2.0e7),
            ),
            # Complex decay rate
            (
                np.log10(1.0e6),
                np.log10(1.0e7),
            ),
            # Complex angular frequency
            (
                np.log10(5.0e7),
                np.log10(1.2e8),
            ),
        ],
        maxiter=250,
        popsize=15,
        tol=1.0e-9,
        polish=True,
        seed=5678,
        workers=1,
    )

    fitted_real_pole = result.real_poles[0]
    fitted_complex_pole = result.complex_poles[0]

    fitted_real_residue = result.residue_fit.residues[0]
    fitted_complex_residue = result.residue_fit.residues[1]

    np.testing.assert_allclose(
        fitted_real_pole,
        expected_real_pole,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        fitted_complex_pole,
        expected_complex_pole,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        fitted_real_residue,
        expected_real_residue,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        fitted_complex_residue,
        expected_complex_residue,
        rtol=1.0e-3,
    )

    assert result.objective_value < 1.0e-8

def test_evolutionary_fit_checks_number_of_bounds():
    with np.testing.assert_raises(ValueError):
        fit_poles_evolutionary(
            frequencies=[1.0e6],
            impedance=[1.0 + 1.0j],
            number_real_poles=1,
            number_complex_pairs=1,
            parameter_bounds=[(6.0, 8.0)],
        )


def test_evolutionary_fit_recovers_mixed_poles_from_finite_wake():
    frequencies = np.linspace(1.0e6, 3.0e7, 500)
    wake_length = 50.0

    expected_real_pole = -1.0e7
    expected_complex_pole = -3.0e6 + 8.0e7j

    expected_real_residue = 8.0e9
    expected_complex_residue = 2.0e10 + 0.5e10j

    poles = np.array(
        [
            expected_real_pole,
            expected_complex_pole,
            np.conj(expected_complex_pole),
        ]
    )

    residues = np.array(
        [
            expected_real_residue,
            expected_complex_residue,
            np.conj(expected_complex_residue),
        ]
    )

    finite_impedance = PoleResidue.finite_wake_impedance(
        frequencies=frequencies,
        poles=poles,
        residues=residues,
        wake_length=wake_length,
    )

    result = fit_poles_evolutionary(
        frequencies=frequencies,
        impedance=finite_impedance,
        number_real_poles=1,
        number_complex_pairs=1,
        parameter_bounds=[
            # Real decay rate
            (
                np.log10(5.0e6),
                np.log10(2.0e7),
            ),
            # Complex decay rate
            (
                np.log10(1.0e6),
                np.log10(1.0e7),
            ),
            # Complex angular frequency
            (
                np.log10(5.0e7),
                np.log10(1.2e8),
            ),
        ],
        wake_length=wake_length,
        maxiter=300,
        popsize=15,
        tol=1.0e-9,
        polish=True,
        seed=9012,
        workers=1,
    )

    np.testing.assert_allclose(
        result.real_poles[0],
        expected_real_pole,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        result.complex_poles[0],
        expected_complex_pole,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        result.residue_fit.residues[0],
        expected_real_residue,
        rtol=1.0e-3,
    )

    np.testing.assert_allclose(
        result.residue_fit.residues[1],
        expected_complex_residue,
        rtol=1.0e-3,
    )

    assert result.objective_value < 1.0e-8