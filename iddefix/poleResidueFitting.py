"""Least-squares fitting for fixed real and complex poles."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .poleResidueFormulas import SPEED_OF_LIGHT


ArrayLike = npt.ArrayLike


@dataclass
class ResidueFitResult:
    """Result of a linear residue fit."""

    poles: np.ndarray
    residues: np.ndarray
    fitted_impedance: np.ndarray
    squared_error: float
    rank: int

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

    design_matrix = np.column_stack(columns)

    real_system_matrix = np.vstack(
        [design_matrix.real, design_matrix.imag]
    )

    real_right_hand_side = np.concatenate(
        [impedance.real, impedance.imag]
    )

    coefficients, _, rank, _ = np.linalg.lstsq(
        real_system_matrix,
        real_right_hand_side,
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

    fitted_impedance = full_basis @ full_residues

    squared_error = float(
        np.sum(np.abs(impedance - fitted_impedance) ** 2)
    )

    return ResidueFitResult(
        poles=full_poles,
        residues=full_residues,
        fitted_impedance=fitted_impedance,
        squared_error=squared_error,
        rank=int(rank),
    )

def pole_objective(
    parameters: ArrayLike,
    frequencies: ArrayLike,
    impedance: ArrayLike,
    number_real_poles: int,
    number_complex_pairs: int,
    wake_length: float | None = None,
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
        )
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

    normalization = np.sum(np.abs(impedance) ** 2)

    if normalization == 0.0:
        normalization = 1.0

    normalized_error = (
        result.squared_error / normalization
    )

    if not np.isfinite(normalized_error):
        return np.inf

    return float(normalized_error)