"""Analytical impedance and wake functions in pole-residue form."""

import numpy as np
import numpy.typing as npt


ArrayLike = npt.ArrayLike

SPEED_OF_LIGHT = 299_792_458.0

class PoleResidue:
    """Evaluate impedance and wake functions from poles and residues."""

    @staticmethod
    def impedance(
        frequencies: ArrayLike,
        poles: ArrayLike,
        residues: ArrayLike,
        direct_term: complex = 0.0,
    ) -> np.ndarray:
        r"""Evaluate

        Z(s) = d + sum_k r_k / (s - p_k),

        with s = 2j*pi*f.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        poles = np.atleast_1d(np.asarray(poles, dtype=complex))
        residues = np.atleast_1d(np.asarray(residues, dtype=complex))

        if poles.size != residues.size:
            raise ValueError("poles and residues must have the same length")

        s = 2j * np.pi * frequencies

        return direct_term + np.sum(
            residues[None, :] / (s[:, None] - poles[None, :]),
            axis=1,
        )

    @staticmethod
    def finite_wake_impedance(
        frequencies: ArrayLike,
        poles: ArrayLike,
        residues: ArrayLike,
        wake_length: float,
        direct_term: complex = 0.0,
    ) -> np.ndarray:
        r"""Evaluate the impedance obtained from a finite wake.

        The model is

        Z_T(s) = sum_k r_k * (1 - exp(-(s - p_k) * T)) / (s - p_k),

        with s = 2j*pi*f and T = wake_length/c.

        Parameters
        ----------
        frequencies
            Frequencies in Hz.
        poles
            Poles in rad/s.
        residues
            Residues corresponding to the poles.
        wake_length
            Simulated wake length in metres.

        Returns
        -------
        numpy.ndarray
            Complex finite-wake impedance.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        poles = np.atleast_1d(np.asarray(poles, dtype=complex))
        residues = np.atleast_1d(np.asarray(residues, dtype=complex))

        if poles.size != residues.size:
            raise ValueError("poles and residues must have the same length")

        if wake_length < 0.0:
            raise ValueError("wake_length must be non-negative")

        s = 2j * np.pi * frequencies
        duration = wake_length / SPEED_OF_LIGHT
        denominator = s[:, None] - poles[None, :]

        basis = (
            -np.expm1(-denominator * duration)
            / denominator
        )

        return direct_term + np.sum(
            residues[None, :] * basis,
            axis=1,
        )



    
    @staticmethod
    def wake(
        times: ArrayLike,
        poles: ArrayLike,
        residues: ArrayLike,
    ) -> np.ndarray:
        r"""Evaluate

        W(t) = sum_k r_k exp(p_k*t),  t >= 0.
        """
        times = np.atleast_1d(np.asarray(times, dtype=float))
        poles = np.atleast_1d(np.asarray(poles, dtype=complex))
        residues = np.atleast_1d(np.asarray(residues, dtype=complex))

        if poles.size != residues.size:
            raise ValueError("poles and residues must have the same length")

        wake = np.sum(
            residues[None, :]
            * np.exp(times[:, None] * poles[None, :]),
            axis=1,
        )

        wake[times < 0.0] = 0.0

        return np.real_if_close(wake)

