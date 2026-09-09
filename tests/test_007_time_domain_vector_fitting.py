import numpy as np
from scipy.optimize import linear_sum_assignment

from iddefix.timeDomainVectorFitting import (
    recursive_exponential_convolution,
    time_domain_vector_fit,
    _build_real_conjugate_basis,
    _restore_conjugate_residues,
    _canonicalize_conjugate_poles,
    _sigma_zeros_from_real_coefficients
)

import pytest

def match_poles(reference_poles, fitted_poles):
    distances = np.abs(
        fitted_poles[:, None]
        - reference_poles[None, :]
    )

    fitted_indices, reference_indices = (
        linear_sum_assignment(distances)
    )

    ordered_poles = np.empty_like(reference_poles)
    ordered_poles[reference_indices] = fitted_poles[
        fitted_indices
    ]

    return ordered_poles


def test_time_domain_vector_fitting_single_pair():
    times = np.linspace(
        0.0,
        5.0e-9,
        2001,
    )
    time_step = times[1] - times[0]

    # Gaussian excitation
    input_signal = np.exp(
        -0.5
        * (
            (times - 0.5e-9)
            / 0.12e-9
        )
        ** 2
    )

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
        output_signal += (
            residue
            * recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
        )

    output_signal = np.real_if_close(
        output_signal
    ).real

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

    pole_error = np.abs(
        fitted_poles - true_poles
    ) / np.abs(true_poles)

    output_error = (
        np.linalg.norm(
            result.fitted_output - output_signal
        )
        / np.linalg.norm(output_signal)
    )

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

    canonical_poles = (
        _canonicalize_conjugate_poles(
            poles,
            relative_tolerance=1.0e-2,
        )
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
        _canonicalize_conjugate_poles(
            poles
        )

def test_real_conjugate_basis_matches_complex_sum():
    times = np.linspace(
        0.0,
        10.0e-9,
        1001,
    )

    time_step = (
        times[1] - times[0]
    )

    signal = np.exp(
        -0.5
        * (
            (times - 1.0e-9)
            / 0.2e-9
        ) ** 2
    )

    positive_pole = (
        -2.0e8
        + 1j * 2.0e9
    )

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

    residue = (
        3.0e8
        - 1j * 4.0e8
    )

    residues = np.array(
        [
            2.0e8,
            residue,
            np.conj(residue),
        ],
        dtype=complex,
    )

    complex_output = (
        filtered_signals @ residues
    )

    real_basis = (
        _build_real_conjugate_basis(
            filtered_signals,
            poles,
        )
    )

    real_coefficients = np.array(
        [
            residues[0].real,
            residue.real,
            residue.imag,
        ]
    )

    real_output = (
        real_basis @ real_coefficients
    )

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
    positive_pole = (
        -2.0e8
        + 1j * 3.0e9
    )

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

    residues = (
        _restore_conjugate_residues(
            coefficients,
            poles,
        )
    )

    assert residues[0].imag == 0.0

    assert residues[1] == (
        2.0 - 3.0j
    )

    assert residues[2] == np.conj(
        residues[1]
    )

def test_tdvf_result_has_exact_conjugate_structure():
    times = np.linspace(
        0.0,
        5.0e-9,
        2001,
    )

    time_step = (
        times[1] - times[0]
    )

    input_signal = np.exp(
        -0.5
        * (
            (times - 0.5e-9)
            / 0.12e-9
        ) ** 2
    )

    true_positive_pole = (
        -8.0e7
        + 1j
        * 2.0
        * np.pi
        * 1.1e9
    )

    true_poles = np.array(
        [
            true_positive_pole,
            np.conj(
                true_positive_pole
            ),
        ]
    )

    true_positive_residue = (
        5.0e10
        + 2.0e10j
    )

    true_residues = np.array(
        [
            true_positive_residue,
            np.conj(
                true_positive_residue
            ),
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
        output_signal += (
            residue
            * recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
        )

    initial_positive_pole = (
        -3.0e8
        + 1j
        * 2.0
        * np.pi
        * 0.9e9
    )

    initial_poles = np.array(
        [
            initial_positive_pole,
            np.conj(
                initial_positive_pole
            ),
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

    assert result.poles[1] == np.conj(
        result.poles[0]
    )

    assert result.residues[1] == np.conj(
        result.residues[0]
    )

    assert result.direct_term.imag == 0.0

    assert result.proportional_term.imag == 0.0

    np.testing.assert_allclose(
        result.fitted_output.imag,
        0.0,
        atol=1.0e-12,
    )

def test_real_sigma_zero_matrix_matches_complex_form():
    positive_pole = (
        -2.0e8
        + 1j * 3.0e9
    )

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

    sigma_residues = (
        _restore_conjugate_residues(
            real_sigma_coefficients,
            poles,
        )
    )

    complex_zero_matrix = (
        np.diag(poles)
        - np.outer(
            np.ones(
                poles.size,
                dtype=complex,
            ),
            sigma_residues,
        )
    )

    complex_zeros = np.linalg.eigvals(
        complex_zero_matrix
    )

    real_form_zeros = (
        _sigma_zeros_from_real_coefficients(
            poles,
            real_sigma_coefficients,
        )
    )

    # Match the two unordered eigenvalue sets.
    distances = np.abs(
        complex_zeros[:, None]
        - real_form_zeros[None, :]
    )

    complex_indices, real_indices = (
        linear_sum_assignment(
            distances
        )
    )

    np.testing.assert_allclose(
        complex_zeros[complex_indices],
        real_form_zeros[real_indices],
        rtol=1.0e-12,
        atol=1.0e-6,
    )