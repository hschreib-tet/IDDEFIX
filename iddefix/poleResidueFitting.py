"""Least-squares fitting for fixed real and complex poles."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from functools import partial

from scipy.optimize import differential_evolution

from .poleResidueFormulas import SPEED_OF_LIGHT


ArrayLike = npt.ArrayLike


@dataclass
class ResidueFitResult:
    """Result of a linear residue fit."""

    poles: np.ndarray
    residues: np.ndarray
    direct_term: float
    fitted_impedance: np.ndarray
    squared_error: float
    weighted_squared_error: float
    rank: int


@dataclass
class PoleOptimizationResult:
    """Result of the evolutionary pole optimization."""

    pole_parameters: np.ndarray
    real_poles: np.ndarray
    complex_poles: np.ndarray
    residue_fit: ResidueFitResult
    objective_value: float
    success: bool
    message: str

def decode_log_poles(
    parameters: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert logarithmic parameters into stable poles.

    The parameter order is

    [log10(a_real),
     log10(alpha_complex),
     log10(beta_complex)],

    where

    p_real = -a

    and

    p_complex = -alpha + 1j*beta.

    All rates are expressed in rad/s.
    """
    parameters = np.asarray(parameters, dtype=float)

    expected_size = (
        number_real_poles
        + 2 * number_complex_pairs
    )

    if parameters.size != expected_size:
        raise ValueError(
            f"expected {expected_size} pole parameters, "
            f"received {parameters.size}"
        )

    real_stop = number_real_poles
    decay_stop = real_stop + number_complex_pairs

    real_rates = 10.0 ** parameters[:real_stop]

    complex_decay_rates = 10.0 ** parameters[
        real_stop:decay_stop
    ]

    complex_frequencies = 10.0 ** parameters[
        decay_stop:
    ]

    real_poles = -np.sort(real_rates).astype(complex)

    complex_poles = (
        -complex_decay_rates
        + 1j * complex_frequencies
    )

    complex_poles = complex_poles[
        np.argsort(complex_poles.imag)
    ]

    return real_poles, complex_poles

def build_fit_weights(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    amplitude_weighting: str = "uniform",
    frequency_weighting: str = "samples",
    magnitude_floor: float | None = None,
) -> np.ndarray:
    """Construct amplitude and frequency-grid fit weights."""
    frequencies = np.atleast_1d(
        np.asarray(frequencies, dtype=float)
    )

    impedance = np.atleast_1d(
        np.asarray(impedance, dtype=complex)
    )

    if frequencies.size != impedance.size:
        raise ValueError(
            "frequencies and impedance must have the same length"
        )

    if frequencies.size == 0:
        raise ValueError("input data must not be empty")

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies must be finite")

    if not np.all(np.isfinite(impedance)):
        raise ValueError("impedance must be finite")

    # Amplitude weighting
    magnitude = np.abs(impedance)

    if magnitude_floor is None:
        maximum_magnitude = np.max(magnitude)
        magnitude_floor = max(
            maximum_magnitude * 1.0e-12,
            np.finfo(float).tiny,
        )

    if magnitude_floor <= 0.0:
        raise ValueError("magnitude_floor must be positive")

    safe_magnitude = np.maximum(
        magnitude,
        magnitude_floor,
    )

    if amplitude_weighting == "uniform":
        amplitude_weights = np.ones_like(frequencies)

    elif amplitude_weighting == "relative":
        amplitude_weights = 1.0 / safe_magnitude

    elif amplitude_weighting == "sqrt_relative":
        amplitude_weights = 1.0 / np.sqrt(safe_magnitude)

    else:
        raise ValueError(
            "amplitude_weighting must be 'uniform', "
            "'relative', or 'sqrt_relative'"
        )

    # Frequency-grid weighting
    if frequency_weighting == "samples":
        frequency_weights = np.ones_like(frequencies)

    else:
        if frequencies.size < 2:
            raise ValueError(
                "frequency-grid weighting requires at least two points"
            )

        if frequency_weighting == "linear":
            coordinate = frequencies

        elif frequency_weighting == "log":
            if np.any(frequencies <= 0.0):
                raise ValueError(
                    "logarithmic frequency weighting requires "
                    "strictly positive frequencies"
                )

            coordinate = np.log(frequencies)

        else:
            raise ValueError(
                "frequency_weighting must be 'samples', "
                "'linear', or 'log'"
            )

        order = np.argsort(coordinate)
        sorted_coordinate = coordinate[order]

        differences = np.diff(sorted_coordinate)

        if np.any(differences <= 0.0):
            raise ValueError(
                "frequencies must not contain duplicates"
            )

        quadrature_weights = np.empty_like(sorted_coordinate)

        quadrature_weights[0] = differences[0] / 2.0
        quadrature_weights[-1] = differences[-1] / 2.0

        if frequencies.size > 2:
            quadrature_weights[1:-1] = (
                sorted_coordinate[2:]
                - sorted_coordinate[:-2]
            ) / 2.0

        # Remove irrelevant overall scaling.
        quadrature_weights /= np.mean(quadrature_weights)

        sorted_frequency_weights = np.sqrt(
            quadrature_weights
        )

        frequency_weights = np.empty_like(
            sorted_frequency_weights
        )

        frequency_weights[order] = sorted_frequency_weights

    weights = amplitude_weights * frequency_weights

    # Improve numerical scaling without changing relative weights.
    weights /= np.sqrt(np.mean(weights**2))

    return weights


def _pole_basis(
    frequencies: np.ndarray,
    poles: np.ndarray,
    wake_length: float | None,
) -> np.ndarray:
    """Construct the impedance basis for individual poles."""
    s = 2j * np.pi * frequencies
    denominator = s[:, None] - poles[None, :]

    if wake_length is None:
        return 1.0 / denominator

    duration = wake_length / SPEED_OF_LIGHT

    return (
        -np.expm1(-denominator * duration)
        / denominator
    )


def fit_residues(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    real_poles: ArrayLike,
    complex_poles: ArrayLike,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    fit_direct_term: bool = False,
) -> ResidueFitResult:
    """Determine optimal residues for fixed poles.

    ``complex_poles`` contains only the poles in the upper half-plane.
    Their complex conjugates are added automatically.
    """
    frequencies = np.atleast_1d(
        np.asarray(frequencies, dtype=float)
    )
    impedance = np.atleast_1d(
        np.asarray(impedance, dtype=complex)
    )
    real_poles = np.atleast_1d(
        np.asarray(real_poles, dtype=complex)
    )
    complex_poles = np.atleast_1d(
        np.asarray(complex_poles, dtype=complex)
    )

    if frequencies.size != impedance.size:
        raise ValueError(
            "frequencies and impedance must have the same length"
        )
    
    if weights is None:
        weights = np.ones_like(frequencies)
    else:
        weights = np.atleast_1d(
            np.asarray(weights, dtype=float)
        )

        if weights.size != frequencies.size:
            raise ValueError(
                "weights and frequencies must have the same length"
            )

        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")

        if np.any(weights <= 0.0):
            raise ValueError("weights must be positive")

    
    if np.any(np.abs(real_poles.imag) > 1.0e-12):
        raise ValueError("real_poles must be real")

    if np.any(complex_poles.imag <= 0.0):
        raise ValueError(
            "complex_poles must lie in the upper half-plane"
        )

    independent_poles = np.concatenate(
        [real_poles.real, complex_poles]
    )

    if independent_poles.size == 0:
        raise ValueError("at least one pole must be provided")

    if np.any(independent_poles.real >= 0.0):
        raise ValueError("all poles must be stable")

    columns = []

    if real_poles.size:
        real_basis = _pole_basis(
            frequencies,
            real_poles,
            wake_length,
        )

        columns.extend(
            real_basis[:, index]
            for index in range(real_poles.size)
        )

    if complex_poles.size:
        positive_basis = _pole_basis(
            frequencies,
            complex_poles,
            wake_length,
        )

        negative_basis = _pole_basis(
            frequencies,
            np.conj(complex_poles),
            wake_length,
        )

        for index in range(complex_poles.size):
            phi_positive = positive_basis[:, index]
            phi_negative = negative_basis[:, index]

            # Coefficient multiplying Re(residue)
            columns.append(phi_positive + phi_negative)

            # Coefficient multiplying Im(residue)
            columns.append(
                1j * (phi_positive - phi_negative)
            )

    if fit_direct_term:
        columns.append(
            np.ones(frequencies.size, dtype=complex)
        )

    design_matrix = np.column_stack(columns)

    real_system_matrix = np.vstack(
        [design_matrix.real, design_matrix.imag]
    )

    real_right_hand_side = np.concatenate(
        [impedance.real, impedance.imag]
    )

    stacked_weights = np.concatenate(
        [weights, weights]
    )

    weighted_system_matrix = (
        stacked_weights[:, None] * real_system_matrix
    )

    weighted_right_hand_side = (
        stacked_weights * real_right_hand_side
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_system_matrix,
        weighted_right_hand_side,
        rcond=None,
    )

    number_real = real_poles.size

    fitted_real_residues = coefficients[:number_real]

    fitted_complex_residues = []

    offset = number_real

    for index in range(complex_poles.size):
        real_part = coefficients[offset + 2 * index]
        imaginary_part = coefficients[offset + 2 * index + 1]

        fitted_complex_residues.append(
            real_part + 1j * imaginary_part
        )

    fitted_complex_residues = np.asarray(
        fitted_complex_residues,
        dtype=complex,
    )

    if fit_direct_term:
        direct_term = float(coefficients[-1])
    else:
        direct_term = 0.0

    full_poles = np.concatenate(
        [
            real_poles,
            complex_poles,
            np.conj(complex_poles),
        ]
    )

    full_residues = np.concatenate(
        [
            fitted_real_residues,
            fitted_complex_residues,
            np.conj(fitted_complex_residues),
        ]
    )

    full_basis = _pole_basis(
        frequencies,
        full_poles,
        wake_length,
    )

    fitted_impedance = (
        direct_term + full_basis @ full_residues
    )

    squared_error = float(
        np.sum(np.abs(impedance - fitted_impedance) ** 2)
    )

    weighted_squared_error = float(
        np.sum(
            np.abs(
                weights * (impedance - fitted_impedance)
            ) ** 2
        )
    )

    return ResidueFitResult(
        poles=full_poles,
        residues=full_residues,
        direct_term=direct_term,
        fitted_impedance=fitted_impedance,
        squared_error=squared_error,
        weighted_squared_error=weighted_squared_error,
        rank=int(rank),
    )

def pole_objective(
    parameters: ArrayLike,
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    wake_length: float | None = None,
    weights: ArrayLike | None = None,
    fit_direct_term: bool = False,
) -> float:
    """Evaluate the normalized fitting error for candidate poles."""
    impedance = np.atleast_1d(
        np.asarray(impedance, dtype=complex)
    )

    real_poles, complex_poles = decode_log_poles(
        parameters,
        number_real_poles,
        number_complex_pairs,
    )

    try:
        result = fit_residues(
            frequencies=frequencies,
            impedance=impedance,
            real_poles=real_poles,
            complex_poles=complex_poles,
            wake_length=wake_length,
            weights=weights,
            fit_direct_term=fit_direct_term,
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.inf
    
    if weights is None:
        weights_array = np.ones_like(
            impedance,
            dtype=float,
        )
    else:
        weights_array = np.asarray(
            weights,
            dtype=float,
        )

    normalization = np.sum(
        np.abs(weights_array * impedance) ** 2
    )

    if normalization == 0.0:
        normalization = 1.0

    normalized_error = (
        result.weighted_squared_error / normalization
    )

    if not np.isfinite(normalized_error):
        return np.inf

    return float(normalized_error)

def fit_poles_evolutionary(
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    parameter_bounds: list[tuple[float, float]],
    wake_length: float | None = None,
    maxiter: int = 1000,
    popsize: int = 15,
    mutation: float | tuple[float, float] = (0.1, 0.5),
    crossover_rate: float = 0.8,
    tol: float = 0.01,
    polish: bool = False,
    seed: int | None = None,
    workers: int = 1,
    amplitude_weighting: str = "uniform",
    frequency_weighting: str = "samples",
    magnitude_floor: float | None = None,
    fit_direct_term: bool = False,
) -> PoleOptimizationResult:
    """Fit pole locations using Differential Evolution.

    Differential Evolution optimizes logarithmic pole parameters.
    For every candidate pole set, the optimal residues are obtained
    by linear least squares.

    Parameters
    ----------
    frequencies
        Frequencies in Hz.
    impedance
        Complex impedance data.
    number_real_poles
        Number of independent negative real poles.
    number_complex_pairs
        Number of complex-conjugate pole pairs.
    parameter_bounds
        Bounds for the logarithmic pole parameters. The order is

        [log10(real decay rates),
         log10(complex decay rates),
         log10(complex angular frequencies)].

        All rates are expressed in rad/s.
    wake_length
        Simulated wake length in metres. If None, fully decayed
        impedance data are assumed.
    """
    frequencies = np.atleast_1d(
        np.asarray(frequencies, dtype=float)
    )

    impedance = np.atleast_1d(
        np.asarray(impedance, dtype=complex)
    )

    weights = build_fit_weights(
        frequencies=frequencies,
        impedance=impedance,
        amplitude_weighting=amplitude_weighting,
        frequency_weighting=frequency_weighting,
        magnitude_floor=magnitude_floor,
    )
    expected_number_bounds = (
        number_real_poles
        + 2 * number_complex_pairs
    )

    if len(parameter_bounds) != expected_number_bounds:
        raise ValueError(
            f"expected {expected_number_bounds} parameter bounds, "
            f"received {len(parameter_bounds)}"
        )

    objective_function = partial(
        pole_objective,
        frequencies=frequencies,
        impedance=impedance,
        number_real_poles=number_real_poles,
        number_complex_pairs=number_complex_pairs,
        wake_length=wake_length,
        weights=weights,
        fit_direct_term=fit_direct_term,
    )

    updating = "immediate" if workers == 1 else "deferred"

    optimization = differential_evolution(
        objective_function,
        bounds=parameter_bounds,
        strategy="rand1bin",
        maxiter=maxiter,
        popsize=popsize,
        mutation=mutation,
        recombination=crossover_rate,
        tol=tol,
        polish=polish,
        seed=seed,
        init="latinhypercube",
        workers=workers,
        updating=updating,
    )

    real_poles, complex_poles = decode_log_poles(
        optimization.x,
        number_real_poles,
        number_complex_pairs,
    )

    residue_fit = fit_residues(
        frequencies=frequencies,
        impedance=impedance,
        real_poles=real_poles,
        complex_poles=complex_poles,
        wake_length=wake_length,
        weights=weights,
        fit_direct_term=fit_direct_term,
    )

    return PoleOptimizationResult(
        pole_parameters=optimization.x,
        real_poles=real_poles,
        complex_poles=complex_poles,
        residue_fit=residue_fit,
        objective_value=float(optimization.fun),
        success=bool(optimization.success),
        message=str(optimization.message),
    )