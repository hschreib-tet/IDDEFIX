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
        raise ValueError(
            "poles must be one-dimensional."
        )

    if not np.all(np.isfinite(poles)):
        raise ValueError(
            "poles must contain only finite values."
        )

    if relative_tolerance < 0.0:
        raise ValueError(
            "relative_tolerance must be non-negative."
        )

    if absolute_tolerance < 0.0:
        raise ValueError(
            "absolute_tolerance must be non-negative."
        )

    pole_scales = np.maximum(
        np.abs(poles),
        1.0,
    )

    real_mask = (
        np.abs(poles.imag)
        <= (
            absolute_tolerance
            + relative_tolerance
            * pole_scales
        )
    )

    real_poles = (
        poles[real_mask].real
        .astype(complex)
    )

    complex_poles = poles[
        ~real_mask
    ]

    positive_poles = complex_poles[
        complex_poles.imag > 0.0
    ]

    negative_poles = complex_poles[
        complex_poles.imag < 0.0
    ]

    if positive_poles.size != negative_poles.size:
        raise ValueError(
            "Complex poles must occur in conjugate pairs."
        )

    canonical_pairs = []

    if positive_poles.size:
        conjugate_distances = np.abs(
            positive_poles[:, None]
            - np.conj(
                negative_poles[None, :]
            )
        )

        positive_indices, negative_indices = (
            linear_sum_assignment(
                conjugate_distances
            )
        )

        for (
            positive_index,
            negative_index,
        ) in zip(
            positive_indices,
            negative_indices,
            strict=True,
        ):
            positive_pole = positive_poles[
                positive_index
            ]

            negative_pole = negative_poles[
                negative_index
            ]

            distance = abs(
                positive_pole
                - np.conj(negative_pole)
            )

            scale = max(
                abs(positive_pole),
                abs(negative_pole),
                1.0,
            )

            tolerance = (
                absolute_tolerance
                + relative_tolerance * scale
            )

            if distance > tolerance:
                raise ValueError(
                    "Complex poles must occur in "
                    "conjugate pairs. "
                    f"Could not pair {positive_pole} "
                    f"with {negative_pole}."
                )

            averaged_positive_pole = (
                0.5
                * (
                    positive_pole
                    + np.conj(negative_pole)
                )
            )

            # Make the orientation of the pair unambiguous.
            averaged_positive_pole = (
                averaged_positive_pole.real
                + 1j
                * abs(
                    averaged_positive_pole.imag
                )
            )

            canonical_pairs.extend(
                [
                    averaged_positive_pole,
                    np.conj(
                        averaged_positive_pole
                    ),
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
        canonical_pairs = (
            canonical_pairs.reshape(-1)
        )
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
        raise ValueError(
            "filtered_signals must be two-dimensional."
        )

    if filtered_signals.shape[1] != poles.size:
        raise ValueError(
            "The number of filtered-signal columns must "
            "equal the number of poles."
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
                "A complex pair must begin with its "
                "positive-imaginary pole."
            )

        if pole_index + 1 >= poles.size:
            raise ValueError(
                "A complex pole is missing its "
                "conjugate partner."
            )

        conjugate_pole = poles[
            pole_index + 1
        ]

        if conjugate_pole != np.conj(pole):
            raise ValueError(
                "Complex poles must be stored as adjacent "
                "exact conjugate pairs."
            )

        positive_column = filtered_signals[
            :,
            pole_index,
        ]

        negative_column = filtered_signals[
            :,
            pole_index + 1,
        ]

        real_residue_column = (
            positive_column
            + negative_column
        ).real

        imaginary_residue_column = (
            1j
            * (
                positive_column
                - negative_column
            )
        ).real

        columns.extend(
            [
                real_residue_column,
                imaginary_residue_column,
            ]
        )

        pole_index += 2

    return np.column_stack(
        columns
    )

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
        raise ValueError(
            "real_coefficients must be one-dimensional."
        )

    if real_coefficients.size != poles.size:
        raise ValueError(
            "There must be one real coefficient per pole."
        )

    residues = np.empty(
        poles.size,
        dtype=complex,
    )

    pole_index = 0
    coefficient_index = 0

    while pole_index < poles.size:
        pole = poles[pole_index]

        if pole.imag == 0.0:
            residues[pole_index] = (
                real_coefficients[
                    coefficient_index
                ]
            )

            pole_index += 1
            coefficient_index += 1
            continue

        if (
            pole.imag < 0.0
            or pole_index + 1 >= poles.size
            or poles[pole_index + 1]
            != np.conj(pole)
        ):
            raise ValueError(
                "Complex poles must be stored as adjacent "
                "exact conjugate pairs."
            )

        residue = (
            real_coefficients[
                coefficient_index
            ]
            + 1j
            * real_coefficients[
                coefficient_index + 1
            ]
        )

        residues[pole_index] = residue
        residues[pole_index + 1] = (
            np.conj(residue)
        )

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
        raise ValueError(
            "poles must be one-dimensional."
        )

    if real_sigma_coefficients.ndim != 1:
        raise ValueError(
            "real_sigma_coefficients must be "
            "one-dimensional."
        )

    if real_sigma_coefficients.size != poles.size:
        raise ValueError(
            "There must be one real sigma coefficient "
            "per pole."
        )

    if not np.all(np.isfinite(poles)):
        raise ValueError(
            "poles must contain only finite values."
        )

    if not np.all(
        np.isfinite(
            real_sigma_coefficients
        )
    ):
        raise ValueError(
            "real_sigma_coefficients must contain "
            "only finite values."
        )

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

    output_vector = (
        real_sigma_coefficients.copy()
    )

    pole_index = 0

    while pole_index < number_of_poles:
        pole = poles[pole_index]

        if pole.imag == 0.0:
            real_pole_matrix[
                pole_index,
                pole_index,
            ] = pole.real

            input_vector[
                pole_index
            ] = 1.0

            pole_index += 1
            continue

        if pole.imag < 0.0:
            raise ValueError(
                "A complex pair must begin with its "
                "positive-imaginary pole."
            )

        if pole_index + 1 >= number_of_poles:
            raise ValueError(
                "A complex pole is missing its "
                "conjugate partner."
            )

        conjugate_pole = poles[
            pole_index + 1
        ]

        if conjugate_pole != np.conj(pole):
            raise ValueError(
                "Complex poles must be stored as adjacent "
                "exact conjugate pairs."
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
        input_vector[
            pole_index
        ] = 2.0

        input_vector[
            pole_index + 1
        ] = 0.0

        pole_index += 2

    # The zeros of sigma(s) are the eigenvalues of
    #
    #     A - b c^T.
    #
    # All quantities in this representation are real.
    zero_matrix = (
        real_pole_matrix
        - np.outer(
            input_vector,
            output_vector,
        )
    )

    return np.linalg.eigvals(
        zero_matrix
    ).astype(complex)

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

        H(s) = direct_term
               + proportional_term * s
               + sum(residue_n / (s - pole_n)).

    Real input and output signals are assumed. Real poles receive
    real residues, while complex poles and residues occur in exact
    complex-conjugate pairs.

    Parameters
    ----------
    times:
        Equidistant time samples.
    input_signal:
        Real excitation x(t).
    output_signal:
        Real response y(t).
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

    Returns
    -------
    TimeDomainVectorFitResult
        Fitted poles, residues, direct and proportional terms and
        the reconstructed output signal.
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
        raise ValueError(
            "times must be one-dimensional."
        )

    if input_signal.shape != times.shape:
        raise ValueError(
            "input_signal and times must have equal shapes."
        )

    if output_signal.shape != times.shape:
        raise ValueError(
            "output_signal and times must have equal shapes."
        )

    minimum_samples = (
        3 if fit_proportional_term else 2
    )

    if times.size < minimum_samples:
        raise ValueError(
            f"At least {minimum_samples} time samples "
            "are required."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            "times must contain only finite values."
        )

    if not np.all(np.isfinite(input_signal)):
        raise ValueError(
            "input_signal must contain only finite values."
        )

    if not np.all(np.isfinite(output_signal)):
        raise ValueError(
            "output_signal must contain only finite values."
        )

    if (
        not isinstance(
            maximum_iterations,
            (int, np.integer),
        )
        or maximum_iterations < 1
    ):
        raise ValueError(
            "maximum_iterations must be a positive integer."
        )

    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
    ):
        raise ValueError(
            "tolerance must be non-negative and finite."
        )

    # -----------------------------------------------------------------
    # Require real input and output signals
    # -----------------------------------------------------------------

    input_scale = max(
        np.max(np.abs(input_signal)),
        np.finfo(float).eps,
    )

    output_scale = max(
        np.max(np.abs(output_signal)),
        np.finfo(float).eps,
    )

    relative_imaginary_input = (
        np.max(np.abs(input_signal.imag))
        / input_scale
    )

    relative_imaginary_output = (
        np.max(np.abs(output_signal.imag))
        / output_scale
    )

    if relative_imaginary_input > 1.0e-10:
        raise ValueError(
            "The real conjugate-pair formulation requires "
            "a real input signal."
        )

    if relative_imaginary_output > 1.0e-10:
        raise ValueError(
            "The real conjugate-pair formulation requires "
            "a real output signal."
        )

    # From this point onward, the least-squares problems are
    # formulated entirely using real quantities.
    input_signal = input_signal.real
    output_signal = output_signal.real

    # -----------------------------------------------------------------
    # Validate equidistant time sampling
    # -----------------------------------------------------------------

    time_steps = np.diff(
        times
    )

    time_step = time_steps[0]

    if (
        not np.isfinite(time_step)
        or time_step <= 0.0
    ):
        raise ValueError(
            "times must be strictly increasing."
        )

    if not np.allclose(
        time_steps,
        time_step,
        rtol=1.0e-10,
        atol=0.0,
    ):
        raise ValueError(
            "The first TD-VF version requires "
            "equidistant time samples."
        )

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

    # Normalize the weights without changing the minimizer.
    weights = (
        weights
        / np.sqrt(
            np.mean(weights**2)
        )
    )

    # -----------------------------------------------------------------
    # Validate and canonicalize starting poles
    # -----------------------------------------------------------------

    poles = (
        _canonicalize_conjugate_poles(
            initial_poles
        )
    )

    if poles.size == 0:
        raise ValueError(
            "initial_poles must be non-empty."
        )

    if np.any(poles.real >= 0.0):
        raise ValueError(
            "All initial poles must lie in the "
            "left half-plane."
        )

    # -----------------------------------------------------------------
    # Derivative of the excitation for the proportional term
    # -----------------------------------------------------------------

    if fit_proportional_term:
        input_derivative = (
            sampled_time_derivative(
                input_signal,
                time_step,
            ).real
        )
    else:
        input_derivative = None

    # -----------------------------------------------------------------
    # Pole relocation
    # -----------------------------------------------------------------

    relocation_errors = []

    for _ in range(maximum_iterations):
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

        real_filtered_input_basis = (
            _build_real_conjugate_basis(
                filtered_inputs,
                poles,
            )
        )

        real_filtered_output_basis = (
            _build_real_conjugate_basis(
                -filtered_outputs,
                poles,
            )
        )

        # c_inf is an auxiliary numerator coefficient of
        # sigma(s) * H(s). It remains present even when the final
        # physical direct term is disabled.
        relocation_columns = [
            input_signal,
        ]

        if fit_proportional_term:
            relocation_columns.append(
                input_derivative
            )

        relocation_columns.extend(
            [
                real_filtered_input_basis,
                real_filtered_output_basis,
            ]
        )

        relocation_matrix = np.column_stack(
            relocation_columns
        )

        weighted_relocation_matrix = (
            weights[:, None]
            * relocation_matrix
        )

        weighted_output_signal = (
            weights
            * output_signal
        )

        relocation_coefficients = (
            _scaled_least_squares(
                weighted_relocation_matrix,
                weighted_output_signal,
            )
        )

        # The last N real coefficients describe the residues of
        # sigma(s). For a complex pair, two consecutive real
        # coefficients represent Re(k) and Im(k).
        sigma_real_coefficients = (
            relocation_coefficients[
                -poles.size:
            ]
        )

        # Calculate the zeros of sigma(s) from the real block
        # representation described in Gustavsen Appendix B.
        relocated_poles = (
            _sigma_zeros_from_real_coefficients(
                poles=poles,
                real_sigma_coefficients=(
                    sigma_real_coefficients
                ),
            )
        )

        if enforce_stability:
            relocated_poles = (
                _stabilize_poles(
                    relocated_poles
                )
            )

        # Numerical eigensolvers preserve conjugacy only up to
        # floating-point accuracy. Project the result back onto
        # an exact real/conjugate structure.
        relocated_poles = (
            _canonicalize_conjugate_poles(
                relocated_poles
            )
        )

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
        relocated_poles = (
            _canonicalize_conjugate_poles(
                relocated_poles
            )
        )

        denominator = max(
            np.linalg.norm(poles),
            np.finfo(float).eps,
        )

        relocation_error = (
            np.linalg.norm(
                relocated_poles - poles
            )
            / denominator
        )

        relocation_errors.append(
            float(relocation_error)
        )

        poles = relocated_poles

        if relocation_error < tolerance:
            break

    # -----------------------------------------------------------------
    # Final residue-identification step
    # -----------------------------------------------------------------

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

    real_filtered_input_basis = (
        _build_real_conjugate_basis(
            filtered_inputs,
            poles,
        )
    )

    residue_columns = []

    if fit_direct_term:
        residue_columns.append(
            input_signal
        )

    if fit_proportional_term:
        residue_columns.append(
            input_derivative
        )

    residue_columns.append(
        real_filtered_input_basis
    )

    residue_matrix = np.column_stack(
        residue_columns
    )

    weighted_residue_matrix = (
        weights[:, None]
        * residue_matrix
    )

    weighted_output_signal = (
        weights
        * output_signal
    )

    residue_coefficients = (
        _scaled_least_squares(
            weighted_residue_matrix,
            weighted_output_signal,
        )
    )

    coefficient_index = 0

    # -----------------------------------------------------------------
    # Extract direct and proportional terms
    # -----------------------------------------------------------------

    if fit_direct_term:
        direct_term = complex(
            residue_coefficients[
                coefficient_index
            ]
        )

        coefficient_index += 1
    else:
        direct_term = 0.0 + 0.0j

    if fit_proportional_term:
        proportional_term = complex(
            residue_coefficients[
                coefficient_index
            ]
        )

        coefficient_index += 1
    else:
        proportional_term = 0.0 + 0.0j

    # -----------------------------------------------------------------
    # Restore real and complex-conjugate residues
    # -----------------------------------------------------------------

    residues = (
        _restore_conjugate_residues(
            real_coefficients=(
                residue_coefficients[
                    coefficient_index:
                ]
            ),
            poles=poles,
        )
    )

    # -----------------------------------------------------------------
    # Reconstruct the fitted output
    # -----------------------------------------------------------------

    fitted_output = (
        direct_term * input_signal
        + filtered_inputs @ residues
    )

    if fit_proportional_term:
        fitted_output += (
            proportional_term
            * input_derivative
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