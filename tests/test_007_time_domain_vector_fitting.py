import numpy as np
from scipy.optimize import linear_sum_assignment

from iddefix.timeDomainVectorFitting import (
    recursive_exponential_convolution,
    time_domain_vector_fit,
)


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