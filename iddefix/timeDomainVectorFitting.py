"""Minimal scalar implementation of Time-Domain Vector Fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment


@dataclass
class TimeDomainVectorFitResult:
    """Result of a scalar Time-Domain Vector Fitting calculation."""

    poles: NDArray[np.complex128]
    residues: NDArray[np.complex128]
    direct_term: complex
    fitted_output: NDArray[np.complex128]
    relocation_errors: list[float]
    iterations: int


def recursive_exponential_convolution(
    signal: ArrayLike,
    pole: complex,
    time_step: float,
) -> NDArray[np.complex128]:
    r"""Convolve a sampled signal with an exponential kernel.

    Computes

        z(t) = integral_0^t exp(pole * (t - tau)) signal(tau) d tau

    using a first-order recursive update and a piecewise-constant
    approximation of the signal over each time interval.
    """

    signal = np.asarray(signal, dtype=complex)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if time_step <= 0.0:
        raise ValueError("time_step must be positive.")

    filtered_signal = np.zeros(
        signal.size,
        dtype=complex,
    )

    decay = np.exp(pole * time_step)

    if abs(pole) > np.finfo(float).eps:
        input_factor = np.expm1(pole * time_step) / pole
    else:
        input_factor = time_step

    for index in range(1, signal.size):
        filtered_signal[index] = (
            decay * filtered_signal[index - 1]
            + input_factor * signal[index]
        )

    return filtered_signal


def _scaled_least_squares(
    system_matrix: NDArray[np.complex128],
    right_hand_side: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Solve a least-squares problem after scaling its columns."""

    column_norms = np.linalg.norm(
        system_matrix,
        axis=0,
    )

    column_norms[column_norms == 0.0] = 1.0

    scaled_matrix = system_matrix / column_norms

    scaled_coefficients, _, _, _ = np.linalg.lstsq(
        scaled_matrix,
        right_hand_side,
        rcond=None,
    )

    return scaled_coefficients / column_norms


def _match_poles(
    old_poles: NDArray[np.complex128],
    new_poles: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Match relocated poles to the previous poles."""

    distances = np.abs(
        new_poles[:, None] - old_poles[None, :]
    )

    new_indices, old_indices = linear_sum_assignment(
        distances
    )

    ordered_poles = np.empty_like(old_poles)
    ordered_poles[old_indices] = new_poles[new_indices]

    return ordered_poles


def _stabilize_poles(
    poles: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Reflect unstable poles into the left half-plane."""

    stabilized_poles = poles.copy()

    unstable = stabilized_poles.real > 0.0

    stabilized_poles[unstable] = (
        -stabilized_poles[unstable].real
        + 1j * stabilized_poles[unstable].imag
    )

    return stabilized_poles


def time_domain_vector_fit(
    times: ArrayLike,
    input_signal: ArrayLike,
    output_signal: ArrayLike,
    initial_poles: ArrayLike,
    maximum_iterations: int = 20,
    tolerance: float = 1.0e-8,
    enforce_stability: bool = True,
) -> TimeDomainVectorFitResult:
    r"""Fit a scalar rational model to time-domain input/output data.

    The identified transfer function is

        H(s) = direct_term
               + sum(residue_n / (s - pole_n)).

    Parameters
    ----------
    times:
        Equidistant time samples.
    input_signal:
        Excitation x(t).
    output_signal:
        Response y(t).
    initial_poles:
        Starting poles for TD-VF pole relocation.
    maximum_iterations:
        Maximum number of pole-relocation iterations.
    tolerance:
        Relative pole movement used as convergence criterion.
    enforce_stability:
        Reflect poles with positive real part into the stable
        left half-plane.
    """

    times = np.asarray(times, dtype=float)
    input_signal = np.asarray(
        input_signal,
        dtype=complex,
    )
    output_signal = np.asarray(
        output_signal,
        dtype=complex,
    )
    poles = np.asarray(
        initial_poles,
        dtype=complex,
    ).copy()

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    if input_signal.shape != times.shape:
        raise ValueError(
            "input_signal and times must have equal shapes."
        )

    if output_signal.shape != times.shape:
        raise ValueError(
            "output_signal and times must have equal shapes."
        )

    if poles.ndim != 1 or poles.size == 0:
        raise ValueError(
            "initial_poles must be a non-empty one-dimensional array."
        )

    if times.size < 2:
        raise ValueError("At least two time samples are required.")

    time_steps = np.diff(times)
    time_step = time_steps[0]

    if not np.allclose(
        time_steps,
        time_step,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise ValueError(
            "The first TD-VF version requires equidistant time samples."
        )

    if np.any(poles.real >= 0.0):
        raise ValueError(
            "All initial poles must lie in the left half-plane."
        )

    relocation_errors = []

    for iteration in range(maximum_iterations):
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

        filtered_outputs = np.column_stack(
            [
                recursive_exponential_convolution(
                    output_signal,
                    pole,
                    time_step,
                )
                for pole in poles
            ]
        )

        # Equation (5) in Grivet-Talocia:
        #
        # y + sum(k_n y_n)
        #     = c_inf x + sum(c_n x_n)
        #
        # Rearranged into A theta = y:
        #
        # [x, x_1, ..., x_N, -y_1, ..., -y_N]
        # [c_inf, c_1, ..., c_N, k_1, ..., k_N]^T = y

        relocation_matrix = np.column_stack(
            [
                input_signal,
                filtered_inputs,
                -filtered_outputs,
            ]
        )

        relocation_coefficients = _scaled_least_squares(
            relocation_matrix,
            output_signal,
        )

        sigma_residues = relocation_coefficients[
            1 + poles.size :
        ]

        relocation_matrix_poles = (
            np.diag(poles)
            - np.outer(
                np.ones(poles.size),
                sigma_residues,
            )
        )

        relocated_poles = np.linalg.eigvals(
            relocation_matrix_poles
        )

        if enforce_stability:
            relocated_poles = _stabilize_poles(
                relocated_poles
            )

        relocated_poles = _match_poles(
            poles,
            relocated_poles,
        )

        denominator = max(
            np.linalg.norm(poles),
            np.finfo(float).eps,
        )

        relocation_error = (
            np.linalg.norm(relocated_poles - poles)
            / denominator
        )

        relocation_errors.append(
            float(relocation_error)
        )

        poles = relocated_poles

        if relocation_error < tolerance:
            break

    # Final residue-identification step, corresponding to
    # equation (7) in the paper.

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

    residue_matrix = np.column_stack(
        [
            input_signal,
            filtered_inputs,
        ]
    )

    residue_coefficients = _scaled_least_squares(
        residue_matrix,
        output_signal,
    )

    direct_term = residue_coefficients[0]
    residues = residue_coefficients[1:]

    fitted_output = (
        direct_term * input_signal
        + filtered_inputs @ residues
    )

    return TimeDomainVectorFitResult(
        poles=poles,
        residues=residues,
        direct_term=direct_term,
        fitted_output=fitted_output,
        relocation_errors=relocation_errors,
        iterations=len(relocation_errors),
    )