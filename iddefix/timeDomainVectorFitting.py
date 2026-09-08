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
    proportional_term: complex = 0.0 + 0.0j


def sampled_time_derivative(
    signal: ArrayLike,
    time_step: float,
) -> NDArray[np.complex128]:
    """Calculate the derivative of an equidistant sampled signal."""

    signal = np.asarray(signal, dtype=complex)

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    if signal.size < 3:
        raise ValueError(
            "At least three samples are required to calculate "
            "the time derivative."
        )

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError(
            "time_step must be a positive finite number."
        )

    return np.gradient(
        signal,
        time_step,
        edge_order=2,
    )


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
        rcond=1.0e-10,
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
    weights: ArrayLike | None = None,
    fit_direct_term: bool = True,
    fit_proportional_term: bool = False,
) -> TimeDomainVectorFitResult:
    r"""Fit a scalar rational model to time-domain input/output data.

    The identified transfer function is

        H(s) = direct_term + proportional_term * s
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
    weights:
        Optional non-negative weights for the time samples.
    fit_direct_term:
        Include the constant direct-coupling term.
    fit_proportional_term:
        Include the proportional term ``proportional_term * s``.
        In time domain this contributes
        ``proportional_term * derivative(input_signal)``.
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
    if weights is None:
        weights = np.ones(times.size, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != times.shape:
            raise ValueError(
                "weights and times must have equal shapes."
            )

        if np.any(~np.isfinite(weights)):
            raise ValueError(
                "weights must contain only finite values."
            )

        if np.any(weights < 0.0):
            raise ValueError(
                "weights must be non-negative."
            )

        if not np.any(weights > 0.0):
            raise ValueError(
                "At least one weight must be positive."
            )

    # Normalize without changing the minimizer.
    weights = weights / np.sqrt(
        np.mean(weights**2)
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

    if fit_proportional_term:
        input_derivative = sampled_time_derivative(
            input_signal,
            time_step,
        )
    else:
        input_derivative = None

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

        relocation_columns = [input_signal]

        if fit_proportional_term:
            relocation_columns.append(input_derivative)

        relocation_columns.extend(
            [
                filtered_inputs,
                -filtered_outputs,
            ]
        )

        relocation_matrix = np.column_stack(
            relocation_columns
        )

        weighted_relocation_matrix = (
            weights[:, None] * relocation_matrix
        )

        weighted_output_signal = (
            weights * output_signal
        )

        relocation_coefficients = _scaled_least_squares(
            weighted_relocation_matrix,
            weighted_output_signal,
        )

        sigma_residues = relocation_coefficients[
            -poles.size :
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

    residue_columns = []

    if fit_direct_term:
        residue_columns.append(input_signal)

    if fit_proportional_term:
        residue_columns.append(input_derivative)

    residue_columns.append(filtered_inputs)

    residue_matrix = np.column_stack(
        residue_columns
    )

    weighted_residue_matrix = (
        weights[:, None] * residue_matrix
    )

    weighted_output_signal = (
        weights * output_signal
    )

    residue_coefficients = _scaled_least_squares(
        weighted_residue_matrix,
        weighted_output_signal,
    )

    coefficient_index = 0

    if fit_direct_term:
        direct_term = residue_coefficients[
            coefficient_index
        ]
        coefficient_index += 1
    else:
        direct_term = 0.0 + 0.0j

    if fit_proportional_term:
        proportional_term = residue_coefficients[
            coefficient_index
        ]
        coefficient_index += 1
    else:
        proportional_term = 0.0 + 0.0j

    residues = residue_coefficients[
        coefficient_index:
    ]

    fitted_output = (
        direct_term * input_signal
        + filtered_inputs @ residues
    )

    if fit_proportional_term:
        fitted_output += (
            proportional_term * input_derivative
        )

    return TimeDomainVectorFitResult(
        poles=poles,
        residues=residues,
        direct_term=direct_term,
        fitted_output=fitted_output,
        relocation_errors=relocation_errors,
        iterations=len(relocation_errors),
        proportional_term=proportional_term,
    )


def evaluate_frequency_response(
    frequencies: ArrayLike,
    poles: ArrayLike,
    residues: ArrayLike,
    direct_term: complex = 0.0,
    fourier_sign: int = -1,
    proportional_term: complex = 0.0,
) -> NDArray[np.complex128]:
    r"""Evaluate a pole-residue transfer function in frequency domain.

    The rational transfer function is

        H(s) = direct_term + proportional_term * s
               + sum_k residues[k] / (s - poles[k]).

    Parameters
    ----------
    frequencies:
        Frequencies in Hz.
    poles:
        Continuous-time poles in rad/s.
    residues:
        Residues corresponding to the poles.
    direct_term:
        Constant direct-coupling term.
    fourier_sign:
        Fourier-transform convention:

        -1 corresponds to exp(-j omega t), hence s = +j omega.
        +1 corresponds to exp(+j omega t), hence s = -j omega.
    proportional_term:
        Coefficient of the term that is linear in ``s``.

    Returns
    -------
    NDArray[np.complex128]
        Complex frequency response.
    """

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    residues = np.asarray(
        residues,
        dtype=complex,
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if poles.ndim != 1:
        raise ValueError(
            "poles must be one-dimensional."
        )

    if residues.shape != poles.shape:
        raise ValueError(
            "poles and residues must have equal shapes."
        )

    if fourier_sign not in (-1, 1):
        raise ValueError(
            "fourier_sign must be either -1 or +1."
        )

    angular_frequencies = (
        2.0 * np.pi * frequencies
    )

    # exp(-j omega t) convention -> s = +j omega
    s = (
        -fourier_sign
        * 1j
        * angular_frequencies
    )

    response = (
        np.full(
            frequencies.shape,
            direct_term,
            dtype=complex,
        )
        + proportional_term * s
    )

    for pole, residue in zip(
        poles,
        residues,
    ):
        response += residue / (s - pole)

    return response


def evaluate_partially_decayed_frequency_response(
    frequencies: ArrayLike,
    poles: ArrayLike,
    residues: ArrayLike,
    wake_length: float,
    direct_term: complex = 0.0,
    fourier_sign: int = -1,
    proportional_term: complex = 0.0,
) -> NDArray[np.complex128]:
    r"""Evaluate a finite-window pole-residue transfer function.

    The regular exponential impulse-response terms are truncated at

        T = wake_length / c.

    The direct and proportional terms are concentrated at t=0 and
    therefore remain unchanged.
    """

    from scipy.constants import c as c_light

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    )

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    residues = np.asarray(
        residues,
        dtype=complex,
    )

    wake_length = float(
        wake_length
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if poles.ndim != 1:
        raise ValueError(
            "poles must be one-dimensional."
        )

    if residues.shape != poles.shape:
        raise ValueError(
            "poles and residues must have equal shapes."
        )

    if not np.all(
        np.isfinite(frequencies)
    ):
        raise ValueError(
            "frequencies must contain only finite values."
        )

    if not np.all(
        np.isfinite(poles)
    ):
        raise ValueError(
            "poles must contain only finite values."
        )

    if not np.all(
        np.isfinite(residues)
    ):
        raise ValueError(
            "residues must contain only finite values."
        )

    if (
        not np.isfinite(wake_length)
        or wake_length <= 0.0
    ):
        raise ValueError(
            "wake_length must be positive and finite."
        )

    if fourier_sign not in (-1, 1):
        raise ValueError(
            "fourier_sign must be either -1 or +1."
        )

    angular_frequencies = (
        2.0 * np.pi * frequencies
    )

    # exp(-j omega t) convention:
    #
    # s = +j omega
    s = (
        -fourier_sign
        * 1j
        * angular_frequencies
    )

    wake_time = (
        wake_length / c_light
    )

    response = (
        np.full(
            frequencies.shape,
            direct_term,
            dtype=complex,
        )
        + proportional_term * s
    )

    for pole, residue in zip(
        poles,
        residues,
        strict=True,
    ):
        response += (
            residue
            * (
                1.0
                - np.exp(
                    (pole - s)
                    * wake_time
                )
            )
            / (s - pole)
        )

    return response