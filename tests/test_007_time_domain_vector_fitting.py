import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from iddefix.timeDomainVectorFitting import (
    _build_real_conjugate_basis,
    _canonicalize_conjugate_poles,
    _restore_conjugate_residues,
    _sigma_zeros_from_real_coefficients,
    evaluate_frequency_response,
    evaluate_partially_decayed_frequency_response,
    recursive_exponential_convolution,
    time_domain_vector_fit,
)


def match_poles(reference_poles, fitted_poles):
    distances = np.abs(fitted_poles[:, None] - reference_poles[None, :])

    fitted_indices, reference_indices = linear_sum_assignment(distances)

    ordered_poles = np.empty_like(reference_poles)
    ordered_poles[reference_indices] = fitted_poles[fitted_indices]

    return ordered_poles


def test_time_domain_vector_fitting_single_pair():
    times = np.linspace(
        0.0,
        5.0e-9,
        2001,
    )
    time_step = times[1] - times[0]

    # Gaussian excitation
    input_signal = np.exp(-0.5 * ((times - 0.5e-9) / 0.12e-9) ** 2)

    # Known complex-conjugate pole pair.
    #
    # The decay time is 12.5 ns, while only 5 ns are observed.
    # The response is therefore deliberately not fully decayed.
    true_poles = np.array(
        [
            -8.0e7 + 1j * 2.0 * np.pi * 1.1e9,
            -8.0e7 - 1j * 2.0 * np.pi * 1.1e9,
        ]
    )

    true_residues = np.array(
        [
            5.0e10 + 2.0e10j,
            5.0e10 - 2.0e10j,
        ]
    )

    output_signal = np.zeros(
        times.size,
        dtype=complex,
    )

    for pole, residue in zip(
        true_poles,
        true_residues,
    ):
        output_signal += residue * recursive_exponential_convolution(
            input_signal,
            pole,
            time_step,
        )

    output_signal = np.real_if_close(output_signal).real

    # Deliberately inaccurate initial poles
    initial_poles = np.array(
        [
            -3.0e8 + 1j * 2.0 * np.pi * 0.9e9,
            -3.0e8 - 1j * 2.0 * np.pi * 0.9e9,
        ]
    )

    result = time_domain_vector_fit(
        times=times,
        input_signal=input_signal,
        output_signal=output_signal,
        initial_poles=initial_poles,
        maximum_iterations=20,
        tolerance=1.0e-10,
    )

    fitted_poles = match_poles(
        true_poles,
        result.poles,
    )

    pole_error = np.abs(fitted_poles - true_poles) / np.abs(true_poles)

    output_error = np.linalg.norm(
        result.fitted_output - output_signal
    ) / np.linalg.norm(output_signal)

    assert np.max(pole_error) < 1.0e-5
    assert output_error < 1.0e-8
    assert result.iterations <= 20


def test_canonicalize_conjugate_poles():
    poles = np.array(
        [
            -3.0e8 - 1j * 2.001e9,
            -1.0e8 + 1j * 1.0e-3,
            -3.01e8 + 1j * 1.999e9,
        ],
        dtype=complex,
    )

    canonical_poles = _canonicalize_conjugate_poles(
        poles,
        relative_tolerance=1.0e-2,
    )

    assert canonical_poles.size == 3

    assert canonical_poles[0].imag == 0.0

    assert canonical_poles[1].imag > 0.0

    np.testing.assert_allclose(
        canonical_poles[2],
        np.conj(canonical_poles[1]),
        rtol=0.0,
        atol=0.0,
    )


def test_unpaired_complex_pole_is_rejected():
    poles = np.array(
        [
            -1.0e8,
            -2.0e8 + 1j * 1.0e9,
        ],
        dtype=complex,
    )

    with pytest.raises(
        ValueError,
        match="conjugate pairs",
    ):
        _canonicalize_conjugate_poles(poles)


def test_real_conjugate_basis_matches_complex_sum():
    times = np.linspace(
        0.0,
        10.0e-9,
        1001,
    )

    time_step = times[1] - times[0]

    signal = np.exp(-0.5 * ((times - 1.0e-9) / 0.2e-9) ** 2)

    positive_pole = -2.0e8 + 1j * 2.0e9

    poles = np.array(
        [
            -1.0e8,
            positive_pole,
            np.conj(positive_pole),
        ],
        dtype=complex,
    )

    filtered_signals = np.column_stack(
        [
            recursive_exponential_convolution(
                signal,
                pole,
                time_step,
            )
            for pole in poles
        ]
    )

    residue = 3.0e8 - 1j * 4.0e8

    residues = np.array(
        [
            2.0e8,
            residue,
            np.conj(residue),
        ],
        dtype=complex,
    )

    complex_output = filtered_signals @ residues

    real_basis = _build_real_conjugate_basis(
        filtered_signals,
        poles,
    )

    real_coefficients = np.array(
        [
            residues[0].real,
            residue.real,
            residue.imag,
        ]
    )

    real_output = real_basis @ real_coefficients

    np.testing.assert_allclose(
        real_output,
        complex_output.real,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        complex_output.imag,
        0.0,
        atol=1.0e-8,
    )


def test_restore_conjugate_residues():
    positive_pole = -2.0e8 + 1j * 3.0e9

    poles = np.array(
        [
            -1.0e8,
            positive_pole,
            np.conj(positive_pole),
        ],
        dtype=complex,
    )

    coefficients = np.array(
        [
            5.0,
            2.0,
            -3.0,
        ]
    )

    residues = _restore_conjugate_residues(
        coefficients,
        poles,
    )

    assert residues[0].imag == 0.0

    assert residues[1] == (2.0 - 3.0j)

    assert residues[2] == np.conj(residues[1])


def test_tdvf_result_has_exact_conjugate_structure():
    times = np.linspace(
        0.0,
        5.0e-9,
        2001,
    )

    time_step = times[1] - times[0]

    input_signal = np.exp(-0.5 * ((times - 0.5e-9) / 0.12e-9) ** 2)

    true_positive_pole = -8.0e7 + 1j * 2.0 * np.pi * 1.1e9

    true_poles = np.array(
        [
            true_positive_pole,
            np.conj(true_positive_pole),
        ]
    )

    true_positive_residue = 5.0e10 + 2.0e10j

    true_residues = np.array(
        [
            true_positive_residue,
            np.conj(true_positive_residue),
        ]
    )

    output_signal = np.zeros(
        times.size,
        dtype=complex,
    )

    for pole, residue in zip(
        true_poles,
        true_residues,
        strict=True,
    ):
        output_signal += residue * recursive_exponential_convolution(
            input_signal,
            pole,
            time_step,
        )

    initial_positive_pole = -3.0e8 + 1j * 2.0 * np.pi * 0.9e9

    initial_poles = np.array(
        [
            initial_positive_pole,
            np.conj(initial_positive_pole),
        ]
    )

    result = time_domain_vector_fit(
        times=times,
        input_signal=input_signal,
        output_signal=output_signal.real,
        initial_poles=initial_poles,
        maximum_iterations=20,
        tolerance=1.0e-10,
    )

    assert result.poles[1] == np.conj(result.poles[0])

    assert result.residues[1] == np.conj(result.residues[0])

    assert result.direct_term.imag == 0.0

    assert result.proportional_term.imag == 0.0

    np.testing.assert_allclose(
        result.fitted_output.imag,
        0.0,
        atol=1.0e-12,
    )


def test_real_sigma_zero_matrix_matches_complex_form():
    positive_pole = -2.0e8 + 1j * 3.0e9

    poles = np.array(
        [
            -5.0e7,
            positive_pole,
            np.conj(positive_pole),
        ],
        dtype=complex,
    )

    real_sigma_coefficients = np.array(
        [
            3.0e7,
            1.0e7,
            -2.0e7,
        ]
    )

    sigma_residues = _restore_conjugate_residues(
        real_sigma_coefficients,
        poles,
    )

    complex_zero_matrix = np.diag(poles) - np.outer(
        np.ones(
            poles.size,
            dtype=complex,
        ),
        sigma_residues,
    )

    complex_zeros = np.linalg.eigvals(complex_zero_matrix)

    real_form_zeros = _sigma_zeros_from_real_coefficients(
        poles,
        real_sigma_coefficients,
    )

    # Match the two unordered eigenvalue sets.
    distances = np.abs(complex_zeros[:, None] - real_form_zeros[None, :])

    complex_indices, real_indices = linear_sum_assignment(distances)

    np.testing.assert_allclose(
        complex_zeros[complex_indices],
        real_form_zeros[real_indices],
        rtol=1.0e-12,
        atol=1.0e-6,
    )


def test_multi_response_fit_recovers_common_poles():
    times = np.linspace(
        0.0,
        5.0e-9,
        2001,
    )
    time_step = times[1] - times[0]

    # One excitation is shared by both response channels.
    input_signal = np.exp(-0.5 * ((times - 0.5e-9) / 0.12e-9) ** 2)

    positive_pole = -8.0e7 + 1j * 2.0 * np.pi * 1.1e9
    true_poles = np.array(
        [
            positive_pole,
            np.conj(positive_pole),
        ]
    )

    positive_residues = np.array(
        [
            5.0e10 + 2.0e10j,
            -1.5e10 + 4.0e10j,
        ]
    )
    true_residues = np.column_stack(
        [
            positive_residues,
            np.conj(positive_residues),
        ]
    )
    true_direct_terms = np.array([0.2, -0.4])

    filtered_inputs = np.column_stack(
        [
            recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
            for pole in true_poles
        ]
    )
    output_signal = np.column_stack(
        [
            true_direct_terms[response_index] * input_signal
            + filtered_inputs @ true_residues[response_index]
            for response_index in range(2)
        ]
    ).real

    initial_positive_pole = -3.0e8 + 1j * 2.0 * np.pi * 0.9e9
    initial_poles = np.array(
        [
            initial_positive_pole,
            np.conj(initial_positive_pole),
        ]
    )

    result = time_domain_vector_fit(
        times=times,
        input_signal=input_signal,
        output_signal=output_signal,
        initial_poles=initial_poles,
        maximum_iterations=20,
        tolerance=1.0e-10,
        channel_weights="rms",
    )

    fitted_poles = match_poles(true_poles, result.poles)
    pole_error = np.abs(fitted_poles - true_poles) / np.abs(true_poles)
    output_errors = np.linalg.norm(
        result.fitted_output - output_signal,
        axis=0,
    ) / np.linalg.norm(output_signal, axis=0)

    assert result.residues.shape == (2, 2)
    assert result.direct_term.shape == (2,)
    assert result.proportional_term.shape == (2,)
    assert result.fitted_output.shape == output_signal.shape
    assert np.max(pole_error) < 1.0e-5
    assert np.max(output_errors) < 1.0e-8

    for response_index in range(2):
        assert result.residues[response_index, 1] == np.conj(
            result.residues[response_index, 0]
        )


def test_multi_response_accepts_separate_input_signals():
    times = np.linspace(0.0, 4.0e-9, 1201)
    time_step = times[1] - times[0]

    input_signals = np.column_stack(
        [
            np.exp(-0.5 * ((times - 0.5e-9) / 0.12e-9) ** 2),
            np.exp(-0.5 * ((times - 0.8e-9) / 0.18e-9) ** 2),
        ]
    )

    positive_pole = -1.2e8 + 1j * 2.0 * np.pi * 0.8e9
    true_poles = np.array(
        [
            positive_pole,
            np.conj(positive_pole),
        ]
    )
    positive_residues = np.array(
        [
            3.0e10 + 1.0e10j,
            1.0e10 - 2.5e10j,
        ]
    )

    output_signal = np.empty_like(input_signals)
    for response_index in range(2):
        filtered_input = np.column_stack(
            [
                recursive_exponential_convolution(
                    input_signals[:, response_index],
                    pole,
                    time_step,
                )
                for pole in true_poles
            ]
        )
        residues = np.array(
            [
                positive_residues[response_index],
                np.conj(positive_residues[response_index]),
            ]
        )
        output_signal[:, response_index] = (filtered_input @ residues).real

    result = time_domain_vector_fit(
        times=times,
        input_signal=input_signals,
        output_signal=output_signal,
        initial_poles=true_poles,
        maximum_iterations=5,
        tolerance=1.0e-10,
    )

    relative_error = np.linalg.norm(
        result.fitted_output - output_signal
    ) / np.linalg.norm(output_signal)

    assert relative_error < 1.0e-8


def test_multi_response_validates_channel_weights():
    times = np.linspace(0.0, 1.0, 11)
    input_signal = np.ones(times.size)
    output_signal = np.ones((times.size, 2))
    poles = np.array([-1.0])

    with pytest.raises(ValueError, match="one value per response"):
        time_domain_vector_fit(
            times=times,
            input_signal=input_signal,
            output_signal=output_signal,
            initial_poles=poles,
            maximum_iterations=1,
            channel_weights=np.ones(3),
        )


def test_multi_response_frequency_evaluation_matches_scalar_calls():
    frequencies = np.linspace(0.0, 2.0e9, 101)
    positive_pole = -2.0e8 + 1j * 3.0e9
    poles = np.array(
        [
            positive_pole,
            np.conj(positive_pole),
        ]
    )
    positive_residues = np.array(
        [
            3.0e9 + 1.0e9j,
            -2.0e9 + 4.0e9j,
        ]
    )
    residues = np.column_stack(
        [
            positive_residues,
            np.conj(positive_residues),
        ]
    )
    direct_terms = np.array([0.5, -0.25])
    proportional_terms = np.array([1.0e-11, -2.0e-11])

    multi_fully_decayed = evaluate_frequency_response(
        frequencies,
        poles,
        residues,
        direct_term=direct_terms,
        proportional_term=proportional_terms,
    )
    multi_partially_decayed = evaluate_partially_decayed_frequency_response(
        frequencies,
        poles,
        residues,
        wake_length=2.0,
        direct_term=direct_terms,
        proportional_term=proportional_terms,
    )

    for response_index in range(2):
        scalar_fully_decayed = evaluate_frequency_response(
            frequencies,
            poles,
            residues[response_index],
            direct_term=direct_terms[response_index],
            proportional_term=proportional_terms[response_index],
        )
        scalar_partially_decayed = evaluate_partially_decayed_frequency_response(
            frequencies,
            poles,
            residues[response_index],
            wake_length=2.0,
            direct_term=direct_terms[response_index],
            proportional_term=proportional_terms[response_index],
        )

        np.testing.assert_allclose(
            multi_fully_decayed[:, response_index],
            scalar_fully_decayed,
        )
        np.testing.assert_allclose(
            multi_partially_decayed[:, response_index],
            scalar_partially_decayed,
        )


def test_multi_response_fits_response_specific_polynomial_terms():
    times = np.linspace(0.0, 3.0e-9, 1501)
    time_step = times[1] - times[0]
    input_signal = np.exp(-0.5 * ((times - 0.5e-9) / 0.12e-9) ** 2)
    input_derivative = np.gradient(
        input_signal,
        time_step,
        edge_order=2,
    )

    positive_pole = -1.0e8 + 1j * 2.0 * np.pi * 1.0e9
    poles = np.array(
        [
            positive_pole,
            np.conj(positive_pole),
        ]
    )
    positive_residues = np.array(
        [
            2.0e10 + 1.0e10j,
            -1.0e10 + 3.0e10j,
        ]
    )
    residues = np.column_stack(
        [
            positive_residues,
            np.conj(positive_residues),
        ]
    )
    direct_terms = np.array([0.3, -0.2])
    proportional_terms = np.array([2.0e-11, -1.0e-11])

    filtered_inputs = np.column_stack(
        [
            recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
            for pole in poles
        ]
    )
    output_signal = np.column_stack(
        [
            direct_terms[response_index] * input_signal
            + proportional_terms[response_index] * input_derivative
            + filtered_inputs @ residues[response_index]
            for response_index in range(2)
        ]
    ).real

    result = time_domain_vector_fit(
        times=times,
        input_signal=input_signal,
        output_signal=output_signal,
        initial_poles=poles,
        maximum_iterations=5,
        tolerance=1.0e-10,
        fit_direct_term=True,
        fit_proportional_term=True,
        channel_weights="rms",
    )

    np.testing.assert_allclose(
        result.direct_term.real,
        direct_terms,
        rtol=1.0e-7,
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        result.proportional_term.real,
        proportional_terms,
        rtol=1.0e-7,
        atol=1.0e-20,
    )
    np.testing.assert_allclose(
        result.fitted_output.real,
        output_signal,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
