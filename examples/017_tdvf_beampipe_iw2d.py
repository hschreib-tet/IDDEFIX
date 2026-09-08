"""Apply Time-Domain Vector Fitting to an IW2D beampipe wake.

The longitudinal wake is treated as an impulse response. A part of
the time-domain data is used for fitting and the remaining samples
are retained for validation.

The script compares:

1. the reference and reconstructed wakes,
2. their finite-window FFT impedances,
3. the finite-window impedance with the analytical pole-residue
   impedance of the fully decayed TD-VF model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import iddefix
from iddefix.timeDomainVectorFitting import (
    evaluate_frequency_response,
    recursive_exponential_convolution,
    sampled_time_derivative,
    time_domain_vector_fit,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# Last physical time used by TD-VF.
#
# Set this to None to fit the complete available wake.
FIT_END_CST_TIME = 5.0e-12

NUMBER_OF_REAL_POLES = 4
NUMBER_OF_COMPLEX_PAIRS = 3

MINIMUM_INITIAL_FREQUENCY = 2.0e9
MAXIMUM_INITIAL_FREQUENCY = 500.0e9

MAXIMUM_PLOT_FREQUENCY = 500.0e9

MAXIMUM_ITERATIONS = 50
RELOCATION_TOLERANCE = 1.0e-8


def find_data_file():
    """Find the IW2D wake file in examples/data."""

    data_directory = (
        Path(__file__).resolve().parent
        / "data"
    )

    possible_names = [
        "beampipe_wake_iw2d.txt",
        "006_beampipe_wake_iw2d.txt",
    ]

    for name in possible_names:
        candidate = data_directory / name

        if candidate.exists():
            return candidate

    expected_files = "\n".join(
        str(data_directory / name)
        for name in possible_names
    )

    raise FileNotFoundError(
        "Could not find the IW2D wake file. "
        "Expected one of:\n"
        + expected_files
    )


def create_initial_poles(
    number_of_real_poles,
    number_of_complex_pairs,
    minimum_frequency,
    maximum_frequency,
    minimum_decay_rate,
    maximum_decay_rate,
):
    """Create real poles and complex-conjugate starting poles."""

    poles = []

    if number_of_real_poles > 0:
        real_decay_rates = np.geomspace(
            minimum_decay_rate,
            maximum_decay_rate,
            number_of_real_poles,
        )

        poles.extend(
            -real_decay_rates.astype(complex)
        )

    if number_of_complex_pairs > 0:
        frequencies = np.geomspace(
            minimum_frequency,
            maximum_frequency,
            number_of_complex_pairs,
        )

        # Start with broadband complex poles.
        #
        # alpha = 2 pi f / (2 Q), with approximately Q = 1.
        complex_decay_rates = (
            np.pi * frequencies
        )

        for frequency, decay_rate in zip(
            frequencies,
            complex_decay_rates,
        ):
            pole = (
                -decay_rate
                + 1j * 2.0 * np.pi * frequency
            )

            poles.extend(
                [
                    pole,
                    np.conj(pole),
                ]
            )

    return np.asarray(poles, dtype=complex)


def evaluate_time_domain_model(
    times,
    input_signal,
    poles,
    residues,
    direct_term,
    proportional_term=0.0,
):
    """Evaluate a pole-residue model in the time domain.

    The model is

        y(t) = d x(t) + h dx(t)/dt
               + sum_k r_k [exp(p_k t) * x(t)].
    """

    time_step = times[1] - times[0]

    output_signal = (
        direct_term
        * input_signal.astype(complex)
        + proportional_term
        * sampled_time_derivative(
            input_signal,
            time_step,
        )
    )

    for pole, residue in zip(
        poles,
        residues,
    ):
        filtered_input = (
            recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
        )

        output_signal += (
            residue * filtered_input
        )

    return output_signal


def normalized_l2_error(
    reference,
    approximation,
):
    """Return a normalized L2 error."""

    reference_norm = np.linalg.norm(reference)

    if reference_norm == 0.0:
        return np.nan

    return (
        np.linalg.norm(
            approximation - reference
        )
        / reference_norm
    )


def print_poles(poles):
    """Print fitted poles in an interpretable form."""

    sorted_poles = sorted(
        poles,
        key=lambda pole: (
            abs(pole.imag),
            pole.imag,
        ),
    )

    print()
    print("Fitted poles")
    print("-" * 96)
    print(
        f"{'Index':>5}"
        f"{'Re(p) [1/s]':>22}"
        f"{'f [Hz]':>22}"
        f"{'tau [s]':>22}"
        f"{'Q-like':>20}"
    )
    print("-" * 96)

    for index, pole in enumerate(sorted_poles):
        if pole.real < 0.0:
            decay_time = -1.0 / pole.real
        else:
            decay_time = np.inf

        frequency = (
            pole.imag / (2.0 * np.pi)
        )

        if pole.real != 0.0:
            quality_factor = (
                abs(pole)
                / (2.0 * abs(pole.real))
            )
        else:
            quality_factor = np.inf

        print(
            f"{index:5d}"
            f"{pole.real:22.8e}"
            f"{frequency:22.8e}"
            f"{decay_time:22.8e}"
            f"{quality_factor:20.6e}"
        )

    print("-" * 96)


def main():
    # -----------------------------------------------------------------
    # Load IW2D data
    # -----------------------------------------------------------------

    data_path = find_data_file()

    data = np.loadtxt(
        data_path,
        comments="#",
    )

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            "The IW2D file must contain at least time and Wlong."
        )

    raw_times = data[:, 0]
    raw_longitudinal_wake = data[:, 1]

    # -----------------------------------------------------------------
    # Reconstruct an exactly equidistant time axis
    # -----------------------------------------------------------------

    raw_time_steps = np.diff(raw_times)

    time_step = (
        raw_times[-1] - raw_times[0]
    ) / (raw_times.size - 1)

    maximum_relative_time_step_deviation = (
        np.max(
            np.abs(
                raw_time_steps - time_step
            )
        )
        / abs(time_step)
    )

    cst_times = (
        raw_times[0]
        + np.arange(
            raw_times.size,
            dtype=float,
        )
        * time_step
    )

    # Numerical time axis beginning at zero.
    times = cst_times - cst_times[0]

    wake = raw_longitudinal_wake.copy()

    wake_scale = np.max(np.abs(wake))

    if wake_scale == 0.0:
        raise ValueError(
            "The longitudinal wake is zero."
        )

    normalized_wake = (
        wake / wake_scale
    )

    # -----------------------------------------------------------------
    # Represent the data as an impulse response
    # -----------------------------------------------------------------

    input_signal = np.zeros_like(
        normalized_wake,
        dtype=float,
    )

    impulse_index = np.argmin(
        np.abs(cst_times)
    )

    # The recursive convolution starts at index 1.
    if impulse_index == 0:
        impulse_index = 1

    # Unit-area discrete impulse.
    input_signal[impulse_index] = (
        1.0 / time_step
    )

    impulse_cst_time = (
        cst_times[impulse_index]
    )

    # -----------------------------------------------------------------
    # Select fit and validation intervals
    # -----------------------------------------------------------------

    if FIT_END_CST_TIME is None:
        fit_mask = np.ones(
            times.size,
            dtype=bool,
        )

        fit_end_cst_time = cst_times[-1]
    else:
        fit_end_cst_time = (
            FIT_END_CST_TIME
        )

        fit_mask = (
            cst_times <= fit_end_cst_time
        )

    # Ensure that the impulse is included in the fit.
    if not fit_mask[impulse_index]:
        raise ValueError(
            "The selected fit interval does not include the impulse."
        )

    validation_mask = ~fit_mask

    fit_end_plot_time = (
        fit_end_cst_time - cst_times[0]
    )

    # -----------------------------------------------------------------
    # Initial poles
    # -----------------------------------------------------------------

    complete_time_length = (
        cst_times[-1] - cst_times[0]
    )

    minimum_decay_rate = (
        1.0 / complete_time_length
    )

    maximum_decay_rate = (
        0.25 / time_step
    )

    initial_poles = create_initial_poles(
        number_of_real_poles=NUMBER_OF_REAL_POLES,
        number_of_complex_pairs=NUMBER_OF_COMPLEX_PAIRS,
        minimum_frequency=MINIMUM_INITIAL_FREQUENCY,
        maximum_frequency=MAXIMUM_INITIAL_FREQUENCY,
        minimum_decay_rate=minimum_decay_rate,
        maximum_decay_rate=maximum_decay_rate,
    )

    # -----------------------------------------------------------------
    # Print configuration
    # -----------------------------------------------------------------

    print()
    print("IW2D beampipe TD-VF benchmark")
    print("=" * 76)
    print(f"Data file: {data_path}")
    print(f"Samples: {times.size}")
    print(
        "Raw time interval: "
        f"{raw_times[0] * 1.0e12:.4f} ps to "
        f"{raw_times[-1] * 1.0e12:.4f} ps"
    )
    print(
        f"Reconstructed time step: "
        f"{time_step * 1.0e15:.6f} fs"
    )
    print(
        "Maximum relative raw time-step deviation: "
        f"{maximum_relative_time_step_deviation:.6e}"
    )
    print(
        "Impulse CST time: "
        f"{impulse_cst_time * 1.0e12:.6f} ps"
    )
    print(
        "Fit interval: "
        f"{cst_times[0] * 1.0e12:.4f} ps to "
        f"{fit_end_cst_time * 1.0e12:.4f} ps"
    )

    if np.any(validation_mask):
        print(
            "Validation interval: "
            f"{cst_times[validation_mask][0] * 1.0e12:.4f} ps to "
            f"{cst_times[-1] * 1.0e12:.4f} ps"
        )
    else:
        print("Validation interval: none")

    print(
        f"Real starting poles: "
        f"{NUMBER_OF_REAL_POLES}"
    )
    print(
        f"Complex starting pairs: "
        f"{NUMBER_OF_COMPLEX_PAIRS}"
    )
    print(
        f"Total number of poles: "
        f"{initial_poles.size}"
    )

    # -----------------------------------------------------------------
    # Run TD-VF
    # -----------------------------------------------------------------

    result = time_domain_vector_fit(
        times=times[fit_mask],
        input_signal=input_signal[fit_mask],
        output_signal=normalized_wake[fit_mask],
        initial_poles=initial_poles,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=RELOCATION_TOLERANCE,
        enforce_stability=True,
        weights=None,
        fit_proportional_term=True,
    )

    # -----------------------------------------------------------------
    # Reconstruct the wake over the complete available interval
    # -----------------------------------------------------------------

    reconstructed_normalized_wake = (
        evaluate_time_domain_model(
            times=times,
            input_signal=input_signal,
            poles=result.poles,
            residues=result.residues,
            direct_term=result.direct_term,
            proportional_term=(
                result.proportional_term
            ),
        )
    )

    reconstructed_wake = (
        wake_scale
        * reconstructed_normalized_wake
    )

    continuation_consistency_error = (
        np.linalg.norm(
            reconstructed_normalized_wake[fit_mask]
            - result.fitted_output
        )
        / np.linalg.norm(
            result.fitted_output
        )
    )

    fit_error = normalized_l2_error(
        wake[fit_mask],
        reconstructed_wake.real[fit_mask],
    )

    if np.any(validation_mask):
        validation_error = normalized_l2_error(
            wake[validation_mask],
            reconstructed_wake.real[
                validation_mask
            ],
        )
    else:
        validation_error = np.nan

    imaginary_fraction = (
        np.linalg.norm(
            reconstructed_wake.imag
        )
        / np.linalg.norm(wake)
    )

    print()
    print("TD-VF result")
    print("-" * 76)
    print(
        f"Relocation iterations: "
        f"{result.iterations}"
    )
    print(
        "Last relocation error: "
        f"{result.relocation_errors[-1]:.6e}"
    )
    print(
        "Continuation consistency error: "
        f"{continuation_consistency_error:.6e}"
    )
    print(
        f"Fit-window L2 error: "
        f"{fit_error:.6e}"
    )
    print(
        f"Validation L2 error: "
        f"{validation_error:.6e}"
    )
    print(
        "Relative imaginary output: "
        f"{imaginary_fraction:.6e}"
    )
    print(
        f"Direct term: "
        f"{result.direct_term:.8e}"
    )
    print(
        f"Proportional term: "
        f"{result.proportional_term:.8e}"
    )

    print_poles(result.poles)

    # -----------------------------------------------------------------
    # Time-domain plots
    # -----------------------------------------------------------------

    time_ps = (
        times * 1.0e12
    )

    fit_end_plot_time_ps = (
        fit_end_plot_time * 1.0e12
    )

    pointwise_error = (
        np.abs(
            reconstructed_wake.real - wake
        )
        / wake_scale
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12, 12),
        constrained_layout=True,
    )

    axes[0].plot(
        time_ps,
        wake,
        color="black",
        linewidth=1.3,
        label="IW2D wake",
    )

    axes[0].plot(
        time_ps,
        reconstructed_wake.real,
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF reconstruction",
    )

    axes[0].axvspan(
        time_ps[0],
        fit_end_plot_time_ps,
        color="tab:blue",
        alpha=0.08,
        label="Fit interval",
    )

    axes[0].axvline(
        fit_end_plot_time_ps,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label="End of fit data",
    )

    axes[0].set_title(
        "IW2D longitudinal wake: "
        "TD-VF reconstruction and extrapolation"
    )
    axes[0].set_xlabel(
        "Time relative to start [ps]"
    )
    axes[0].set_ylabel(
        r"$W_\parallel$ [V/C]"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time_ps[fit_mask],
        wake[fit_mask],
        color="black",
        linewidth=1.3,
        label="IW2D wake",
    )

    axes[1].plot(
        time_ps[fit_mask],
        reconstructed_wake.real[fit_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF fit",
    )

    axes[1].set_title(
        "Wake inside the fit interval"
    )
    axes[1].set_xlabel(
        "Time relative to start [ps]"
    )
    axes[1].set_ylabel(
        r"$W_\parallel$ [V/C]"
    )
    axes[1].grid(True)
    axes[1].legend()

    axes[2].semilogy(
        time_ps,
        np.maximum(
            pointwise_error,
            np.finfo(float).eps,
        ),
        color="tab:blue",
    )

    axes[2].axvline(
        fit_end_plot_time_ps,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
    )

    axes[2].set_title(
        "Absolute error normalized by maximum wake"
    )
    axes[2].set_xlabel(
        "Time relative to start [ps]"
    )
    axes[2].set_ylabel(
        r"$|W_\mathrm{fit}-W_\mathrm{IW2D}|/"
        r"\max|W_\mathrm{IW2D}|$"
    )
    axes[2].grid(True, which="both")

    plt.show()

    # -----------------------------------------------------------------
    # Finite-window impedance comparison
    # -----------------------------------------------------------------

    cst_frequencies, iw2d_impedance = (
        iddefix.compute_fft(
            cst_times,
            wake,
        )
    )

    tdvf_frequencies, tdvf_finite_impedance = (
        iddefix.compute_fft(
            cst_times,
            reconstructed_wake.real,
        )
    )

    if not np.allclose(
        cst_frequencies,
        tdvf_frequencies,
    ):
        raise RuntimeError(
            "The IW2D and TD-VF FFT frequency axes differ."
        )

    frequency_mask = (
        cst_frequencies
        <= MAXIMUM_PLOT_FREQUENCY
    )

    # -----------------------------------------------------------------
    # Analytical fully-decayed TD-VF impedance
    #
    # The input is a unit-area discrete impulse. Therefore the
    # identified transfer function itself is the analytical
    # longitudinal impedance, up to the normalization by wake_scale.
    # -----------------------------------------------------------------

    tdvf_fully_decayed_impedance = (
        wake_scale
        * evaluate_frequency_response(
            frequencies=cst_frequencies,
            poles=result.poles,
            residues=result.residues,
            direct_term=result.direct_term,
            fourier_sign=-1,
            proportional_term=(
                result.proportional_term
            ),
        )
    )

    impedance_figure, impedance_axes = (
        plt.subplots(
            3,
            1,
            figsize=(12, 11),
            sharex=True,
            constrained_layout=True,
        )
    )

    frequency_ghz = (
        cst_frequencies[frequency_mask]
        / 1.0e9
    )

    impedance_axes[0].plot(
        frequency_ghz,
        iw2d_impedance.real[frequency_mask],
        color="black",
        linewidth=1.3,
        label="IW2D finite-wake impedance",
    )

    impedance_axes[0].plot(
        frequency_ghz,
        tdvf_finite_impedance.real[
            frequency_mask
        ],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF finite-wake impedance",
    )

    impedance_axes[0].plot(
        frequency_ghz,
        tdvf_fully_decayed_impedance.real[
            frequency_mask
        ],
        color="tab:blue",
        linestyle=":",
        linewidth=1.3,
        label="TD-VF fully-decayed impedance",
    )

    impedance_axes[1].plot(
        frequency_ghz,
        iw2d_impedance.imag[frequency_mask],
        color="black",
        linewidth=1.3,
        label="IW2D finite-wake impedance",
    )

    impedance_axes[1].plot(
        frequency_ghz,
        tdvf_finite_impedance.imag[
            frequency_mask
        ],
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF finite-wake impedance",
    )

    impedance_axes[1].plot(
        frequency_ghz,
        tdvf_fully_decayed_impedance.imag[
            frequency_mask
        ],
        color="tab:blue",
        linestyle=":",
        linewidth=1.3,
        label="TD-VF fully-decayed impedance",
    )

    impedance_axes[2].semilogy(
        frequency_ghz,
        np.maximum(
            np.abs(
                iw2d_impedance[frequency_mask]
            ),
            np.finfo(float).eps,
        ),
        color="black",
        linewidth=1.3,
        label="IW2D finite-wake impedance",
    )

    impedance_axes[2].semilogy(
        frequency_ghz,
        np.maximum(
            np.abs(
                tdvf_finite_impedance[
                    frequency_mask
                ]
            ),
            np.finfo(float).eps,
        ),
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF finite-wake impedance",
    )

    impedance_axes[2].semilogy(
        frequency_ghz,
        np.maximum(
            np.abs(
                tdvf_fully_decayed_impedance[
                    frequency_mask
                ]
            ),
            np.finfo(float).eps,
        ),
        color="tab:blue",
        linestyle=":",
        linewidth=1.3,
        label="TD-VF fully-decayed impedance",
    )

    impedance_axes[0].set_ylabel(
        r"$\operatorname{Re}(Z_\parallel)$ [Ohm]"
    )

    impedance_axes[1].set_ylabel(
        r"$\operatorname{Im}(Z_\parallel)$ [Ohm]"
    )

    impedance_axes[2].set_ylabel(
        r"$|Z_\parallel|$ [Ohm]"
    )

    impedance_axes[2].set_xlabel(
        "Frequency [GHz]"
    )

    for axis in impedance_axes:
        axis.grid(True, which="both")
        axis.legend()

    plt.show()

    # -----------------------------------------------------------------
    # Pole plot
    # -----------------------------------------------------------------

    pole_figure, pole_axis = plt.subplots(
        figsize=(9, 6),
        constrained_layout=True,
    )

    pole_axis.scatter(
        initial_poles.real / 1.0e9,
        initial_poles.imag
        / (2.0 * np.pi * 1.0e9),
        marker="o",
        s=45,
        facecolors="none",
        edgecolors="tab:blue",
        label="Initial poles",
    )

    pole_axis.scatter(
        result.poles.real / 1.0e9,
        result.poles.imag
        / (2.0 * np.pi * 1.0e9),
        marker="x",
        s=65,
        color="tab:orange",
        label="Fitted poles",
    )

    pole_axis.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    pole_axis.set_title(
        "Initial and fitted TD-VF poles"
    )

    pole_axis.set_xlabel(
        r"$\operatorname{Re}(p)$ "
        r"[$10^9$ s$^{-1}$]"
    )

    pole_axis.set_ylabel(
        r"$\operatorname{Im}(p)/(2\pi)$ "
        r"[GHz]"
    )

    pole_axis.grid(True)
    pole_axis.legend()

    plt.show()


if __name__ == "__main__":
    main()
