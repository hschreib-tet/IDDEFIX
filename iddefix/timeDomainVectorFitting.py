"""Scalar and multi-response Time-Domain Vector Fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment


@dataclass
class TimeDomainVectorFitResult:
    """Result of a Time-Domain Vector Fitting calculation.

    Scalar fits retain the historical one-dimensional result shapes.
    For a multi-response fit, residues have shape ``(responses, poles)``,
    fitted_output has shape ``(samples, responses)``, and the direct and
    proportional terms have shape ``(responses,)``.
    """

    poles: NDArray[np.complex128]
    residues: NDArray[np.complex128]
    direct_term: complex | NDArray[np.complex128]
    fitted_output: NDArray[np.complex128]
    relocation_errors: list[float]
    iterations: int
    proportional_term: complex | NDArray[np.complex128] = 0.0 + 0.0j


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
            "At least three samples are required to calculate the time derivative."
        )

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step must be a positive finite number.")

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
            decay * filtered_signal[index - 1] + input_factor * signal[index]
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

    distances = np.abs(new_poles[:, None] - old_poles[None, :])

    new_indices, old_indices = linear_sum_assignment(distances)

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
        -stabilized_poles[unstable].real + 1j * stabilized_poles[unstable].imag
    )

    return stabilized_poles


def _canonicalize_conjugate_poles(
    poles: ArrayLike,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 0.0,
) -> NDArray[np.complex128]:
    """Return real poles and exact complex-conjugate pairs.

    Nearly real poles are projected onto the real axis. Remaining
    complex poles are matched to their conjugates and replaced by
    exactly conjugate averaged pairs.

    The returned ordering is

        real poles,
        positive-imaginary pole,
        corresponding negative-imaginary pole,
        ...

    Parameters
    ----------
    poles:
        One-dimensional array of poles.
    relative_tolerance:
        Relative tolerance used to identify real poles and
        conjugate partners.
    absolute_tolerance:
        Absolute tolerance used in addition to the relative
        tolerance.

    Returns
    -------
    NDArray[np.complex128]
        Canonically ordered poles.

    Raises
    ------
    ValueError
        If a complex pole has no matching conjugate partner.
    """

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    if poles.ndim != 1:
        raise ValueError("poles must be one-dimensional.")

    if not np.all(np.isfinite(poles)):
        raise ValueError("poles must contain only finite values.")

    if relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be non-negative.")

    if absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance must be non-negative.")

    pole_scales = np.maximum(
        np.abs(poles),
        1.0,
    )

    real_mask = np.abs(poles.imag) <= (
        absolute_tolerance + relative_tolerance * pole_scales
    )

    real_poles = poles[real_mask].real.astype(complex)

    complex_poles = poles[~real_mask]

    positive_poles = complex_poles[complex_poles.imag > 0.0]

    negative_poles = complex_poles[complex_poles.imag < 0.0]

    if positive_poles.size != negative_poles.size:
        raise ValueError("Complex poles must occur in conjugate pairs.")

    canonical_pairs = []

    if positive_poles.size:
        conjugate_distances = np.abs(
            positive_poles[:, None] - np.conj(negative_poles[None, :])
        )

        positive_indices, negative_indices = linear_sum_assignment(conjugate_distances)

        for (
            positive_index,
            negative_index,
        ) in zip(
            positive_indices,
            negative_indices,
            strict=True,
        ):
            positive_pole = positive_poles[positive_index]

            negative_pole = negative_poles[negative_index]

            distance = abs(positive_pole - np.conj(negative_pole))

            scale = max(
                abs(positive_pole),
                abs(negative_pole),
                1.0,
            )

            tolerance = absolute_tolerance + relative_tolerance * scale

            if distance > tolerance:
                raise ValueError(
                    "Complex poles must occur in "
                    "conjugate pairs. "
                    f"Could not pair {positive_pole} "
                    f"with {negative_pole}."
                )

            averaged_positive_pole = 0.5 * (positive_pole + np.conj(negative_pole))

            # Make the orientation of the pair unambiguous.
            averaged_positive_pole = averaged_positive_pole.real + 1j * abs(
                averaged_positive_pole.imag
            )

            canonical_pairs.extend(
                [
                    averaged_positive_pole,
                    np.conj(averaged_positive_pole),
                ]
            )

    real_poles = np.asarray(
        sorted(
            real_poles,
            key=lambda pole: pole.real,
            reverse=True,
        ),
        dtype=complex,
    )

    canonical_pairs = np.asarray(
        sorted(
            zip(
                canonical_pairs[0::2],
                canonical_pairs[1::2],
                strict=True,
            ),
            key=lambda pair: pair[0].imag,
        ),
        dtype=complex,
    )

    if canonical_pairs.size:
        canonical_pairs = canonical_pairs.reshape(-1)
    else:
        canonical_pairs = np.empty(
            0,
            dtype=complex,
        )

    return np.concatenate(
        [
            real_poles,
            canonical_pairs,
        ]
    )


def _build_real_conjugate_basis(
    filtered_signals: NDArray[np.complex128],
    poles: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Build a real basis for real and conjugate pole contributions.

    For a real pole, one real column is used.

    For a complex-conjugate pair p and conj(p), the columns are

        phi_p + phi_conj(p)

    and

        1j * (phi_p - phi_conj(p)).

    The corresponding real coefficients are the real and imaginary
    parts of the residue belonging to the positive-imaginary pole.
    """

    filtered_signals = np.asarray(
        filtered_signals,
        dtype=complex,
    )

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    if filtered_signals.ndim != 2:
        raise ValueError("filtered_signals must be two-dimensional.")

    if filtered_signals.shape[1] != poles.size:
        raise ValueError(
            "The number of filtered-signal columns must equal the number of poles."
        )

    columns = []

    pole_index = 0

    while pole_index < poles.size:
        pole = poles[pole_index]

        if pole.imag == 0.0:
            columns.append(
                filtered_signals[
                    :,
                    pole_index,
                ].real
            )

            pole_index += 1
            continue

        if pole.imag < 0.0:
            raise ValueError(
                "A complex pair must begin with its positive-imaginary pole."
            )

        if pole_index + 1 >= poles.size:
            raise ValueError("A complex pole is missing its conjugate partner.")

        conjugate_pole = poles[pole_index + 1]

        if conjugate_pole != np.conj(pole):
            raise ValueError(
                "Complex poles must be stored as adjacent exact conjugate pairs."
            )

        positive_column = filtered_signals[
            :,
            pole_index,
        ]

        negative_column = filtered_signals[
            :,
            pole_index + 1,
        ]

        real_residue_column = (positive_column + negative_column).real

        imaginary_residue_column = (1j * (positive_column - negative_column)).real

        columns.extend(
            [
                real_residue_column,
                imaginary_residue_column,
            ]
        )

        pole_index += 2

    return np.column_stack(columns)


def _restore_conjugate_residues(
    real_coefficients: ArrayLike,
    poles: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Restore real and conjugate residues from real coefficients."""

    real_coefficients = np.asarray(
        real_coefficients,
        dtype=float,
    )

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    if real_coefficients.ndim != 1:
        raise ValueError("real_coefficients must be one-dimensional.")

    if real_coefficients.size != poles.size:
        raise ValueError("There must be one real coefficient per pole.")

    residues = np.empty(
        poles.size,
        dtype=complex,
    )

    pole_index = 0
    coefficient_index = 0

    while pole_index < poles.size:
        pole = poles[pole_index]

        if pole.imag == 0.0:
            residues[pole_index] = real_coefficients[coefficient_index]

            pole_index += 1
            coefficient_index += 1
            continue

        if (
            pole.imag < 0.0
            or pole_index + 1 >= poles.size
            or poles[pole_index + 1] != np.conj(pole)
        ):
            raise ValueError(
                "Complex poles must be stored as adjacent exact conjugate pairs."
            )

        residue = (
            real_coefficients[coefficient_index]
            + 1j * real_coefficients[coefficient_index + 1]
        )

        residues[pole_index] = residue
        residues[pole_index + 1] = np.conj(residue)

        pole_index += 2
        coefficient_index += 2

    return residues


def _sigma_zeros_from_real_coefficients(
    poles: NDArray[np.complex128],
    real_sigma_coefficients: ArrayLike,
) -> NDArray[np.complex128]:
    """Calculate the zeros of sigma from a real state-space matrix.

    The scaling function is

        sigma(s) = 1 + sum_k c_k / (s - p_k).

    Real poles are represented by scalar real blocks. Every
    complex-conjugate pole pair is represented by a real 2 x 2
    block as described by Gustavsen and Semlyen.

    Parameters
    ----------
    poles:
        Canonically ordered poles. Real poles must appear first.
        Complex poles must occur as adjacent exact conjugate pairs,
        with the positive-imaginary pole first.
    real_sigma_coefficients:
        One real coefficient per pole. For a complex pair, two
        consecutive coefficients represent the real and imaginary
        parts of the residue belonging to the positive-imaginary
        pole.

    Returns
    -------
    NDArray[np.complex128]
        Zeros of sigma(s).
    """

    poles = np.asarray(
        poles,
        dtype=complex,
    )

    real_sigma_coefficients = np.asarray(
        real_sigma_coefficients,
        dtype=float,
    )

    if poles.ndim != 1:
        raise ValueError("poles must be one-dimensional.")

    if real_sigma_coefficients.ndim != 1:
        raise ValueError("real_sigma_coefficients must be one-dimensional.")

    if real_sigma_coefficients.size != poles.size:
        raise ValueError("There must be one real sigma coefficient per pole.")

    if not np.all(np.isfinite(poles)):
        raise ValueError("poles must contain only finite values.")

    if not np.all(np.isfinite(real_sigma_coefficients)):
        raise ValueError("real_sigma_coefficients must contain only finite values.")

    number_of_poles = poles.size

    real_pole_matrix = np.zeros(
        (
            number_of_poles,
            number_of_poles,
        ),
        dtype=float,
    )

    input_vector = np.zeros(
        number_of_poles,
        dtype=float,
    )

    output_vector = real_sigma_coefficients.copy()

    pole_index = 0

    while pole_index < number_of_poles:
        pole = poles[pole_index]

        if pole.imag == 0.0:
            real_pole_matrix[
                pole_index,
                pole_index,
            ] = pole.real

            input_vector[pole_index] = 1.0

            pole_index += 1
            continue

        if pole.imag < 0.0:
            raise ValueError(
                "A complex pair must begin with its positive-imaginary pole."
            )

        if pole_index + 1 >= number_of_poles:
            raise ValueError("A complex pole is missing its conjugate partner.")

        conjugate_pole = poles[pole_index + 1]

        if conjugate_pole != np.conj(pole):
            raise ValueError(
                "Complex poles must be stored as adjacent exact conjugate pairs."
            )

        alpha = pole.real
        beta = pole.imag

        # Real 2 x 2 representation of the conjugate pair
        #
        #     alpha + j beta
        #     alpha - j beta
        #
        # in the real basis used in Appendix B.
        real_pole_matrix[
            pole_index : pole_index + 2,
            pole_index : pole_index + 2,
        ] = np.array(
            [
                [alpha, beta],
                [-beta, alpha],
            ],
            dtype=float,
        )

        # Under the same similarity transformation, the original
        # complex input vector [1, 1]^T becomes [2, 0]^T.
        input_vector[pole_index] = 2.0

        input_vector[pole_index + 1] = 0.0

        pole_index += 2

    # The zeros of sigma(s) are the eigenvalues of
    #
    #     A - b c^T.
    #
    # All quantities in this representation are real.
    zero_matrix = real_pole_matrix - np.outer(
        input_vector,
        output_vector,
    )

    return np.linalg.eigvals(zero_matrix).astype(complex)


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
    channel_weights: ArrayLike | str | None = None,
) -> TimeDomainVectorFitResult:
    r"""Fit rational models with a common pole set to time-domain data.

    The identified transfer function is

        H_m(s) = direct_term_m
                 + proportional_term_m * s
                 + sum(residue_mn / (s - pole_n)).

    Real input and output signals are assumed. Real poles receive
    real residues, while complex poles and residues occur in exact
    complex-conjugate pairs.

    Parameters
    ----------
    times:
        Equidistant time samples.
    input_signal:
        Real excitation. A one-dimensional signal is shared by all
        responses. A two-dimensional array must have shape
        ``(number_samples, number_responses)`` and supplies one excitation
        for each response.
    output_signal:
        Real response signal with shape ``(number_samples,)`` or
        ``(number_samples, number_responses)``. All responses share the
        fitted pole set but receive independent residues and polynomial
        terms.
    initial_poles:
        Starting poles for TD-VF pole relocation. Complex poles
        must be supplied in conjugate pairs.
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

            proportional_term * derivative(input_signal).
    channel_weights:
        Optional response-channel weights. ``None`` gives every response
        equal weight. ``"rms"`` scales each response inversely with its RMS
        amplitude. An array supplies one non-negative weight per response.

    Returns
    -------
    TimeDomainVectorFitResult
        Fitted common poles, response-specific residues and polynomial
        terms, and the reconstructed output signal. For scalar input/output
        data, the historical scalar and one-dimensional result shapes are
        retained.
    """

    # -----------------------------------------------------------------
    # Convert input arrays
    # -----------------------------------------------------------------

    times = np.asarray(
        times,
        dtype=float,
    )

    input_signal = np.asarray(
        input_signal,
        dtype=complex,
    )

    output_signal = np.asarray(
        output_signal,
        dtype=complex,
    )

    # -----------------------------------------------------------------
    # Validate dimensions and sample numbers
    # -----------------------------------------------------------------

    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")

    scalar_output = output_signal.ndim == 1

    if scalar_output:
        if output_signal.shape != times.shape:
            raise ValueError("output_signal and times must have equal lengths.")
        output_signal = output_signal[:, None]
    elif output_signal.ndim == 2:
        if output_signal.shape[0] != times.size:
            raise ValueError("The first output_signal dimension must equal times.size.")
        if output_signal.shape[1] == 0:
            raise ValueError("output_signal must contain at least one response.")
    else:
        raise ValueError("output_signal must be one- or two-dimensional.")

    number_responses = output_signal.shape[1]

    if input_signal.ndim == 1:
        if input_signal.shape != times.shape:
            raise ValueError("input_signal and times must have equal lengths.")
        input_signal = np.broadcast_to(
            input_signal[:, None],
            output_signal.shape,
        ).copy()
    elif input_signal.ndim == 2:
        if input_signal.shape != output_signal.shape:
            raise ValueError(
                "A two-dimensional input_signal must have the same shape "
                "as output_signal."
            )
    else:
        raise ValueError("input_signal must be one- or two-dimensional.")

    minimum_samples = 3 if fit_proportional_term else 2

    if times.size < minimum_samples:
        raise ValueError(f"At least {minimum_samples} time samples are required.")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain only finite values.")

    if not np.all(np.isfinite(input_signal)):
        raise ValueError("input_signal must contain only finite values.")

    if not np.all(np.isfinite(output_signal)):
        raise ValueError("output_signal must contain only finite values.")

    if (
        not isinstance(
            maximum_iterations,
            (int, np.integer),
        )
        or maximum_iterations < 1
    ):
        raise ValueError("maximum_iterations must be a positive integer.")

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be non-negative and finite.")

    # -----------------------------------------------------------------
    # Require real input and output signals
    # -----------------------------------------------------------------

    input_scale = np.maximum(
        np.max(np.abs(input_signal), axis=0),
        np.finfo(float).eps,
    )

    output_scale = np.maximum(
        np.max(np.abs(output_signal), axis=0),
        np.finfo(float).eps,
    )

    relative_imaginary_input = np.max(np.abs(input_signal.imag), axis=0) / input_scale

    relative_imaginary_output = (
        np.max(np.abs(output_signal.imag), axis=0) / output_scale
    )

    if np.any(relative_imaginary_input > 1.0e-10):
        raise ValueError(
            "The real conjugate-pair formulation requires a real input signal."
        )

    if np.any(relative_imaginary_output > 1.0e-10):
        raise ValueError(
            "The real conjugate-pair formulation requires a real output signal."
        )

    # From this point onward, the least-squares problems are
    # formulated entirely using real quantities.
    input_signal = input_signal.real
    output_signal = output_signal.real

    # -----------------------------------------------------------------
    # Validate equidistant time sampling
    # -----------------------------------------------------------------

    time_steps = np.diff(times)

    time_step = time_steps[0]

    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("times must be strictly increasing.")

    if not np.allclose(
        time_steps,
        time_step,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise ValueError("The first TD-VF version requires equidistant time samples.")

    # -----------------------------------------------------------------
    # Validate and normalize weights
    # -----------------------------------------------------------------

    if weights is None:
        weights = np.ones(
            times.size,
            dtype=float,
        )
    else:
        weights = np.asarray(
            weights,
            dtype=float,
        )

        if weights.shape != times.shape:
            raise ValueError("weights and times must have equal shapes.")

        if np.any(~np.isfinite(weights)):
            raise ValueError("weights must contain only finite values.")

        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative.")

        if not np.any(weights > 0.0):
            raise ValueError("At least one weight must be positive.")

    # Normalize the weights without changing the minimizer.
    weights = weights / np.sqrt(np.mean(weights**2))

    # -----------------------------------------------------------------
    # Validate and normalize response-channel weights
    # -----------------------------------------------------------------

    if channel_weights is None:
        channel_weights = np.ones(number_responses, dtype=float)
    elif isinstance(channel_weights, str):
        if channel_weights.lower() != "rms":
            raise ValueError("channel_weights must be None, 'rms', or an array.")

        response_rms = np.sqrt(np.mean(output_signal.real**2, axis=0))
        maximum_rms = np.max(response_rms)
        if maximum_rms == 0.0:
            channel_weights = np.ones(number_responses, dtype=float)
        else:
            rms_floor = maximum_rms * 1.0e-12
            informative_response = response_rms > rms_floor
            channel_weights = np.zeros(number_responses, dtype=float)
            channel_weights[informative_response] = (
                1.0 / response_rms[informative_response]
            )
    else:
        channel_weights = np.asarray(channel_weights, dtype=float)

        if channel_weights.shape != (number_responses,):
            raise ValueError("channel_weights must contain one value per response.")
        if np.any(~np.isfinite(channel_weights)):
            raise ValueError("channel_weights must contain only finite values.")
        if np.any(channel_weights < 0.0):
            raise ValueError("channel_weights must be non-negative.")
        if not np.any(channel_weights > 0.0):
            raise ValueError("At least one channel weight must be positive.")

    channel_weights = channel_weights / np.max(channel_weights)
    channel_weights = channel_weights / np.sqrt(np.mean(channel_weights**2))

    # -----------------------------------------------------------------
    # Validate and canonicalize starting poles
    # -----------------------------------------------------------------

    poles = _canonicalize_conjugate_poles(initial_poles)

    if poles.size == 0:
        raise ValueError("initial_poles must be non-empty.")

    if np.any(poles.real >= 0.0):
        raise ValueError("All initial poles must lie in the left half-plane.")

    # -----------------------------------------------------------------
    # Derivative of the excitation for the proportional term
    # -----------------------------------------------------------------

    if fit_proportional_term:
        input_derivative = np.column_stack(
            [
                sampled_time_derivative(
                    input_signal[:, response_index],
                    time_step,
                ).real
                for response_index in range(number_responses)
            ]
        )
    else:
        input_derivative = None

    # -----------------------------------------------------------------
    # Pole relocation
    # -----------------------------------------------------------------

    relocation_errors = []

    for _ in range(maximum_iterations):
        filtered_inputs = []
        filtered_outputs = []

        for response_index in range(number_responses):
            filtered_inputs.append(
                np.column_stack(
                    [
                        recursive_exponential_convolution(
                            input_signal[:, response_index],
                            pole,
                            time_step,
                        )
                        for pole in poles
                    ]
                )
            )
            filtered_outputs.append(
                np.column_stack(
                    [
                        recursive_exponential_convolution(
                            output_signal[:, response_index],
                            pole,
                            time_step,
                        )
                        for pole in poles
                    ]
                )
            )

        # Time-domain pole-relocation equation:
        #
        # y + sum(k_n y_n)
        #     = c_inf x
        #       + h_inf dx/dt
        #       + sum(c_n x_n)
        #
        # Rearranged:
        #
        # [x, dx/dt, x-basis, -y-basis] theta = y
        #
        # For every complex-conjugate pole pair, the two complex
        # columns are replaced by the real columns
        #
        #     phi_p + phi_conj(p)
        #
        # and
        #
        #     1j * (phi_p - phi_conj(p)).
        #
        # The corresponding unknowns are the real and imaginary
        # parts of the residue belonging to the pole with positive
        # imaginary part.

        numerator_size = poles.size + 1 + int(fit_proportional_term)
        relocation_matrix = np.zeros(
            (
                times.size * number_responses,
                numerator_size * number_responses + poles.size,
            ),
            dtype=float,
        )
        relocation_right_hand_side = np.empty(
            times.size * number_responses,
            dtype=float,
        )
        relocation_row_weights = np.empty(
            times.size * number_responses,
            dtype=float,
        )

        for response_index in range(number_responses):
            row_slice = slice(
                response_index * times.size,
                (response_index + 1) * times.size,
            )
            numerator_slice = slice(
                response_index * numerator_size,
                (response_index + 1) * numerator_size,
            )

            real_filtered_input_basis = _build_real_conjugate_basis(
                filtered_inputs[response_index],
                poles,
            )
            real_filtered_output_basis = _build_real_conjugate_basis(
                -filtered_outputs[response_index],
                poles,
            )

            numerator_columns = [input_signal[:, response_index]]
            if fit_proportional_term:
                numerator_columns.append(input_derivative[:, response_index])
            numerator_columns.append(real_filtered_input_basis)

            relocation_matrix[row_slice, numerator_slice] = np.column_stack(
                numerator_columns
            )
            relocation_matrix[row_slice, -poles.size :] = real_filtered_output_basis
            relocation_right_hand_side[row_slice] = output_signal[:, response_index]
            relocation_row_weights[row_slice] = (
                weights * channel_weights[response_index]
            )

        weighted_relocation_matrix = relocation_row_weights[:, None] * relocation_matrix
        weighted_output_signal = relocation_row_weights * relocation_right_hand_side

        relocation_coefficients = _scaled_least_squares(
            weighted_relocation_matrix,
            weighted_output_signal,
        )

        # The last N real coefficients describe the residues of
        # sigma(s). For a complex pair, two consecutive real
        # coefficients represent Re(k) and Im(k).
        sigma_real_coefficients = relocation_coefficients[-poles.size :]

        # Calculate the zeros of sigma(s) from the real block
        # representation described in Gustavsen Appendix B.
        relocated_poles = _sigma_zeros_from_real_coefficients(
            poles=poles,
            real_sigma_coefficients=(sigma_real_coefficients),
        )

        if enforce_stability:
            relocated_poles = _stabilize_poles(relocated_poles)

        # Numerical eigensolvers preserve conjugacy only up to
        # floating-point accuracy. Project the result back onto
        # an exact real/conjugate structure.
        relocated_poles = _canonicalize_conjugate_poles(relocated_poles)

        # Associate new poles with the previous poles before
        # calculating the relative pole movement.
        relocated_poles = _match_poles(
            poles,
            relocated_poles,
        )

        # Restore the canonical ordering:
        #
        # real poles first, followed by adjacent conjugate pairs,
        # with the positive-imaginary pole first.
        relocated_poles = _canonicalize_conjugate_poles(relocated_poles)

        denominator = max(
            np.linalg.norm(poles),
            np.finfo(float).eps,
        )

        relocation_error = np.linalg.norm(relocated_poles - poles) / denominator

        relocation_errors.append(float(relocation_error))

        poles = relocated_poles

        if relocation_error < tolerance:
            break

    # -----------------------------------------------------------------
    # Final residue-identification step
    # -----------------------------------------------------------------

    residues = np.empty(
        (number_responses, poles.size),
        dtype=complex,
    )
    direct_term = np.zeros(number_responses, dtype=complex)
    proportional_term = np.zeros(number_responses, dtype=complex)
    fitted_output = np.empty(output_signal.shape, dtype=complex)

    for response_index in range(number_responses):
        filtered_input = np.column_stack(
            [
                recursive_exponential_convolution(
                    input_signal[:, response_index],
                    pole,
                    time_step,
                )
                for pole in poles
            ]
        )
        real_filtered_input_basis = _build_real_conjugate_basis(
            filtered_input,
            poles,
        )

        residue_columns = []
        if fit_direct_term:
            residue_columns.append(input_signal[:, response_index])
        if fit_proportional_term:
            residue_columns.append(input_derivative[:, response_index])
        residue_columns.append(real_filtered_input_basis)

        residue_matrix = np.column_stack(residue_columns)
        response_weights = weights * channel_weights[response_index]
        residue_coefficients = _scaled_least_squares(
            response_weights[:, None] * residue_matrix,
            response_weights * output_signal[:, response_index],
        )

        coefficient_index = 0
        if fit_direct_term:
            direct_term[response_index] = complex(
                residue_coefficients[coefficient_index]
            )
            coefficient_index += 1
        if fit_proportional_term:
            proportional_term[response_index] = complex(
                residue_coefficients[coefficient_index]
            )
            coefficient_index += 1

        residues[response_index] = _restore_conjugate_residues(
            real_coefficients=residue_coefficients[coefficient_index:],
            poles=poles,
        )
        fitted_output[:, response_index] = (
            direct_term[response_index] * input_signal[:, response_index]
            + filtered_input @ residues[response_index]
        )
        if fit_proportional_term:
            fitted_output[:, response_index] += (
                proportional_term[response_index] * input_derivative[:, response_index]
            )

    if scalar_output:
        residues = residues[0]
        direct_term = complex(direct_term[0])
        proportional_term = complex(proportional_term[0])
        fitted_output = fitted_output[:, 0]

    return TimeDomainVectorFitResult(
        poles=poles,
        residues=residues,
        direct_term=direct_term,
        fitted_output=fitted_output,
        relocation_errors=relocation_errors,
        iterations=len(relocation_errors),
        proportional_term=proportional_term,
    )


def _prepare_frequency_response_coefficients(
    poles: ArrayLike,
    residues: ArrayLike,
    direct_term: ArrayLike,
    proportional_term: ArrayLike,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.complex128],
    bool,
]:
    """Validate scalar or multi-response pole-residue coefficients."""

    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)

    if poles.ndim != 1:
        raise ValueError("poles must be one-dimensional.")

    scalar_response = residues.ndim == 1
    if scalar_response:
        if residues.shape != poles.shape:
            raise ValueError("poles and residues must have equal shapes.")
        residues = residues[None, :]
    elif residues.ndim == 2:
        if residues.shape[1] != poles.size:
            raise ValueError(
                "The last residues dimension must equal the number of poles."
            )
        if residues.shape[0] == 0:
            raise ValueError("residues must contain at least one response.")
    else:
        raise ValueError("residues must be one- or two-dimensional.")

    number_responses = residues.shape[0]

    def prepare_polynomial_term(term, name):
        term = np.asarray(term, dtype=complex)
        if term.ndim == 0:
            return np.full(number_responses, term.item(), dtype=complex)
        if term.shape != (number_responses,):
            raise ValueError(f"{name} must be scalar or have one value per response.")
        return term

    direct_term = prepare_polynomial_term(direct_term, "direct_term")
    proportional_term = prepare_polynomial_term(
        proportional_term,
        "proportional_term",
    )

    return (
        poles,
        residues,
        direct_term,
        proportional_term,
        scalar_response,
    )


def evaluate_frequency_response(
    frequencies: ArrayLike,
    poles: ArrayLike,
    residues: ArrayLike,
    direct_term: ArrayLike = 0.0,
    fourier_sign: int = -1,
    proportional_term: ArrayLike = 0.0,
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

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional.")

    (
        poles,
        residues,
        direct_term,
        proportional_term,
        scalar_response,
    ) = _prepare_frequency_response_coefficients(
        poles,
        residues,
        direct_term,
        proportional_term,
    )

    if fourier_sign not in (-1, 1):
        raise ValueError("fourier_sign must be either -1 or +1.")

    angular_frequencies = 2.0 * np.pi * frequencies

    # exp(-j omega t) convention -> s = +j omega
    s = -fourier_sign * 1j * angular_frequencies

    basis = 1.0 / (s[:, None] - poles[None, :])
    response = (
        direct_term[None, :]
        + s[:, None] * proportional_term[None, :]
        + basis @ residues.T
    )

    return response[:, 0] if scalar_response else response


def evaluate_partially_decayed_frequency_response(
    frequencies: ArrayLike,
    poles: ArrayLike,
    residues: ArrayLike,
    wake_length: float,
    direct_term: ArrayLike = 0.0,
    fourier_sign: int = -1,
    proportional_term: ArrayLike = 0.0,
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

    wake_length = float(wake_length)

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be one-dimensional.")

    (
        poles,
        residues,
        direct_term,
        proportional_term,
        scalar_response,
    ) = _prepare_frequency_response_coefficients(
        poles,
        residues,
        direct_term,
        proportional_term,
    )

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must contain only finite values.")

    if not np.all(np.isfinite(poles)):
        raise ValueError("poles must contain only finite values.")

    if not np.all(np.isfinite(residues)):
        raise ValueError("residues must contain only finite values.")

    if not np.isfinite(wake_length) or wake_length <= 0.0:
        raise ValueError("wake_length must be positive and finite.")

    if fourier_sign not in (-1, 1):
        raise ValueError("fourier_sign must be either -1 or +1.")

    angular_frequencies = 2.0 * np.pi * frequencies

    # exp(-j omega t) convention:
    #
    # s = +j omega
    s = -fourier_sign * 1j * angular_frequencies

    wake_time = wake_length / c_light

    basis = (1.0 - np.exp((poles[None, :] - s[:, None]) * wake_time)) / (
        s[:, None] - poles[None, :]
    )
    response = (
        direct_term[None, :]
        + s[:, None] * proportional_term[None, :]
        + basis @ residues.T
    )

    return response[:, 0] if scalar_response else response
