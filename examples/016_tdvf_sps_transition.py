"""Apply Time-Domain Vector Fitting to the CST SPS-transition wake.

The CST wake potential is treated as the response to a Gaussian bunch.
Only the first part of the transient response is used for fitting.
The remaining CST samples are retained for validation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c as c_light


import iddefix

from iddefix.timeDomainVectorFitting import (
    evaluate_frequency_response,
    recursive_exponential_convolution,
    sampled_time_derivative,
    time_domain_vector_fit,
)


MAXIMUM_ITERATIONS = 50
RELOCATION_TOLERANCE = 1.0e-8

def create_initial_poles(
    number_of_complex_pairs,
    minimum_frequency,
    maximum_frequency,
    complex_damping,
    include_real_pole=True,
    real_pole_location=-5.0e8,
):
    """Create stable starting poles distributed over a frequency band."""

    frequencies = np.linspace(
        minimum_frequency,
        maximum_frequency,
        number_of_complex_pairs,
    )

    poles = []

    if include_real_pole:
        poles.append(complex(real_pole_location))

    for frequency in frequencies:
        pole = (
            -complex_damping
            + 1j * 2.0 * np.pi * frequency
        )

        poles.extend(
            [
                pole,
                np.conj(pole),
            ]
        )

    return np.asarray(poles, dtype=complex)


def evaluate_model(
    times,
    input_signal,
    poles,
    residues,
    direct_term,
    proportional_term=0.0,
):
    """Evaluate a fitted pole-residue model in the time domain."""

    time_step = times[1] - times[0]

    input_derivative = sampled_time_derivative(
    input_signal,
    time_step,
    )

    output_signal = (
        direct_term * input_signal.astype(complex)
        + proportional_term * input_derivative
    )

    for pole, residue in zip(poles, residues):
        output_signal += (
            residue
            * recursive_exponential_convolution(
                input_signal,
                pole,
                time_step,
            )
        )

    return output_signal


def normalized_l2_error(reference, approximation):
    """Calculate the normalized L2 error."""

    denominator = np.linalg.norm(reference)

    if denominator == 0.0:
        return np.nan

    return (
        np.linalg.norm(approximation - reference)
        / denominator
    )


def print_poles(poles):
    """Print fitted poles as damping rate and frequency."""

    print()
    print(
        f"{'Index':>5}"
        f"{'Re(p) [1/s]':>22}"
        f"{'Im(p)/(2pi) [Hz]':>24}"
        f"{'Decay time [ns]':>22}"
    )
    print("-" * 73)

    sorted_poles = sorted(
        poles,
        key=lambda pole: (
            abs(pole.imag),
            pole.imag,
        ),
    )

    for index, pole in enumerate(sorted_poles):
        if pole.real < 0.0:
            decay_time_ns = (
                -1.0 / pole.real * 1.0e9
            )
        else:
            decay_time_ns = np.inf

        frequency = pole.imag / (2.0 * np.pi)

        print(
            f"{index:5d}"
            f"{pole.real:22.8e}"
            f"{frequency:24.8e}"
            f"{decay_time_ns:22.6e}"
        )


def main():
    data_path = (
        Path(__file__).resolve().parent
        / "data"
        / "004_SPS_model_transitions_q26.txt"
    )

    data = np.loadtxt(
        data_path,
        comments="#",
        delimiter="\t",
    )

    # The first column is given in ns.
    original_times = data[:, 0] * 1.0e-9

    # Same transverse normalization as in example 004a.
    original_wake = data[:, 2] * c_light


    # Use the complete signal, including the region before the bunch.
    wake = original_wake.copy()

    # The CST ASCII export contains small rounding deviations.
    # Determine the intended sampling interval robustly.
    time_differences = np.diff(original_times)
    time_step = np.median(time_differences)

    maximum_time_step_deviation = np.max(
        np.abs(time_differences - time_step)
    )

    relative_time_step_deviation = (
        maximum_time_step_deviation / abs(time_step)
    )

    print(
        "Maximum relative time-step deviation: "
        f"{relative_time_step_deviation:.6e}"
    )

    if relative_time_step_deviation > 1.0e-6:
        raise ValueError(
            "The CST time axis is not sufficiently equidistant."
        )

    # Reconstruct an exactly equidistant numerical time axis.
    times = (
        np.arange(original_times.size, dtype=float)
        * time_step
    )

    # Corresponding CST coordinate, including its original offset.
    # This coordinate is needed to position the Gaussian bunch at t = 0.
    cst_times = original_times[0] + times


    wake_scale = np.max(np.abs(wake))

    if wake_scale == 0.0:
        raise ValueError("The selected wake signal is zero.")

    normalized_wake = wake / wake_scale

    # Gaussian bunch profile used as TD-VF excitation.
    bunch_sigma = 1.0e-10  # 0.1 ns

    input_signal = np.exp(
        -0.5 * (cst_times / bunch_sigma) ** 2
    )

    # Its absolute normalization only changes the fitted residues.
    input_signal /= np.max(input_signal)

    # Fit only the first four nanoseconds after the selected start.
    fit_end_cst_time = 8.0e-9

    #fit_end_cst_time = original_times[-1]

    fit_mask = cst_times <= fit_end_cst_time

    # Location of the fit boundary on the shifted plotting axis.
    fit_end_time = (
        fit_end_cst_time - original_times[0]
    )

    # Initial model:
    #   one negative real pole
    #   five complex-conjugate pole pairs
    initial_poles = create_initial_poles(
        number_of_complex_pairs=5,
        minimum_frequency=0.10e9,
        maximum_frequency=5.0e9,
        complex_damping=2.0 * np.pi * 0.15e9,
        include_real_pole=True,
        real_pole_location=-5.0e8,
    )

    print()
    print("SPS transition TD-VF benchmark")
    print("=" * 72)
    print(f"Data file: {data_path}")
    print(
        "Original time interval: "
        f"{original_times[0] * 1.0e9:.4f} ns to "
        f"{original_times[-1] * 1.0e9:.4f} ns"
    )
    print(
        "Selected time interval: "
        f"{times[0] * 1.0e9:.4f} ns to "
        f"{times[-1] * 1.0e9:.4f} ns"
    )
    print(
        f"Time step: {time_step * 1.0e12:.4f} ps"
    )
    print(
        f"Fit interval: 0 to "
        f"{fit_end_time * 1.0e9:.2f} ns"
    )
    print(
        f"Validation interval: "
        f"{fit_end_time * 1.0e9:.2f} to "
        f"{times[-1] * 1.0e9:.2f} ns"
    )
    print(f"Number of poles: {initial_poles.size}")


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

    reconstructed_normalized = evaluate_model(
        times=times,
        input_signal=input_signal,
        poles=result.poles,
        residues=result.residues,
        direct_term=result.direct_term,
        proportional_term=result.proportional_term,
    )

    continuation_consistency_error = (
    np.linalg.norm(
        reconstructed_normalized[fit_mask]
        - result.fitted_output
    )
    / np.linalg.norm(result.fitted_output)
    )

    print(
        "Continuation consistency error: "
        f"{continuation_consistency_error:.6e}"
    )

    reconstructed_wake = (
        reconstructed_normalized * wake_scale
    )

    # CST wake in the original spatial transverse normalization.
    cst_wake_for_fft = (
        wake / c_light
    )

    # TD-VF reconstruction in the same normalization.
    tdvf_wake_for_fft = (
        reconstructed_wake.real / c_light
    )

    cst_frequencies, cst_wake_transform = (
        iddefix.compute_fft(
            cst_times,
            cst_wake_for_fft,
        )
    )

    tdvf_fft_frequencies, tdvf_wake_transform = (
        iddefix.compute_fft(
            cst_times,
            tdvf_wake_for_fft,
        )
    )

    cst_finite_impedance = (
        -1j * cst_wake_transform
    )

    tdvf_finite_impedance = (
        -1j * tdvf_wake_transform
    )

    if not np.allclose(
        cst_frequencies,
        tdvf_fft_frequencies,
    ):
        raise RuntimeError(
            "CST and TD-VF FFT frequency axes differ."
        )

    validation_mask = ~fit_mask

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
        np.linalg.norm(reconstructed_wake.imag)
        / np.linalg.norm(wake)
    )

    print()
    print(f"Relocation iterations: {result.iterations}")
    print(
        "Last relocation error: "
        f"{result.relocation_errors[-1]:.6e}"
    )
    print(f"Fit-window L2 error:    {fit_error:.6e}")
    print(
        f"Validation L2 error:    {validation_error:.6e}"
    )
    print(
        "Relative imaginary output: "
        f"{imaginary_fraction:.6e}"
    )
    print(
        f"Direct term: {result.direct_term:.8e}"
    )

    print(
    f"Proportional term: "
    f"{result.proportional_term:.8e}"
    )

    print_poles(result.poles)

    time_ns = times * 1.0e9

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
        time_ns,
        wake,
        color="black",
        linewidth=1.5,
        label="CST wake",
    )

    axes[0].plot(
        time_ns,
        reconstructed_wake.real,
        color="tab:orange",
        linestyle="--",
        linewidth=1.1,
        label="TD-VF reconstruction",
    )

    axes[0].axvspan(
        0.0,
        fit_end_time * 1.0e9,
        color="tab:blue",
        alpha=0.08,
        label="Fit interval",
    )

    axes[0].axvline(
        fit_end_time * 1.0e9,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
        label="End of fit data",
    )

    axes[0].set_title(
        "SPS-transition CST wake: TD-VF extrapolation"
    )
    axes[0].set_xlabel(
        "Time relative to selected start [ns]"
    )
    axes[0].set_ylabel(
        "Dipolar transverse wake"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time_ns[fit_mask],
        wake[fit_mask],
        color="black",
        linewidth=1.8,
        label="CST wake",
    )

    axes[1].plot(
        time_ns[fit_mask],
        reconstructed_wake.real[fit_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="TD-VF fit",
    )

    axes[1].set_title("Wake inside the fit interval")
    axes[1].set_xlabel(
        "Time relative to selected start [ns]"
    )
    axes[1].set_ylabel(
        "Dipolar transverse wake"
    )
    axes[1].grid(True)
    axes[1].legend()

    axes[2].semilogy(
        time_ns,
        np.maximum(
            pointwise_error,
            np.finfo(float).eps,
        ),
        color="tab:blue",
    )

    axes[2].axvline(
        fit_end_time * 1.0e9,
        color="tab:red",
        linestyle=":",
        linewidth=1.5,
    )

    axes[2].set_title(
        "Absolute error normalized by maximum CST wake"
    )
    axes[2].set_xlabel(
        "Time relative to selected start [ns]"
    )
    axes[2].set_ylabel(
        r"$|W_\mathrm{fit}-W_\mathrm{CST}|/"
        r"\max|W_\mathrm{CST}|$"
    )
    axes[2].grid(True, which="both")

    plt.show()

    pole_figure, pole_axis = plt.subplots(
        figsize=(9, 6),
        constrained_layout=True,
    )

    pole_axis.scatter(
        initial_poles.real / 1.0e9,
        initial_poles.imag
        / (2.0 * np.pi * 1.0e9),
        marker="o",
        s=55,
        facecolors="none",
        edgecolors="tab:blue",
        label="Initial poles",
    )

    pole_axis.scatter(
        result.poles.real / 1.0e9,
        result.poles.imag
        / (2.0 * np.pi * 1.0e9),
        marker="x",
        s=70,
        color="tab:orange",
        label="Fitted poles",
    )

    pole_axis.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    pole_axis.set_title("Initial and fitted TD-VF poles")
    pole_axis.set_xlabel(
        r"$\operatorname{Re}(p)$ [$10^9$ s$^{-1}$]"
    )
    pole_axis.set_ylabel(
        r"$\operatorname{Im}(p)/(2\pi)$ [GHz]"
    )
    pole_axis.grid(True)
    pole_axis.legend()


    plt.show()

        # -------------------------------------------------------------
    # Analytical fully-decayed pole-residue transfer function
    # -------------------------------------------------------------

    tdvf_transfer_function = evaluate_frequency_response(
        frequencies=cst_frequencies,
        poles=result.poles,
        residues=result.residues,
        direct_term=result.direct_term,
        proportional_term=result.proportional_term,
        fourier_sign=-1,
    )

    # The Gaussian input was normalized to a maximum value of one.
    # Its integral converts it into a unit-area bunch distribution.
    input_signal_area = np.trapezoid(
        input_signal,
        cst_times,
    )

    # The fitted output was normalized by wake_scale. In addition,
    # the transverse wake was multiplied by c_light before fitting.
    #
    # Therefore the fully-decayed transverse point-wake impedance is
    #
    #     Z_perp = -j/c * H_point
    #
    # with
    #
    #     H_point = wake_scale * input_signal_area * H_fit.
    tdvf_fully_decayed_impedance = (
        -1j
        * wake_scale
        * input_signal_area
        / c_light
        * tdvf_transfer_function
    )

    # The CST curve is a wake potential produced by the Gaussian bunch.
    # Its spectrum contains the additional Gaussian bunch spectrum.
    angular_frequencies = (
        2.0 * np.pi * cst_frequencies
    )

    gaussian_bunch_spectrum = np.exp(
        -0.5
        * (
            angular_frequencies
            * bunch_sigma
        )
        ** 2
    )

    tdvf_fully_decayed_wake_potential_impedance = (
        tdvf_fully_decayed_impedance
        * gaussian_bunch_spectrum
    )


    maximum_plot_frequency = 5.0e9

    frequency_mask = (
        cst_frequencies <= maximum_plot_frequency
    )

    impedance_figure, impedance_axes = plt.subplots(
        3,
        1,
        figsize=(11, 10),
        sharex=True,
        constrained_layout=True,
    )

    impedance_axes[0].plot(
        cst_frequencies[frequency_mask],
        cst_finite_impedance.real[frequency_mask],
        color="black",
        linewidth=1.5,
        label="CST finite-wake impedance",
    )

    impedance_axes[0].plot(
        cst_frequencies[frequency_mask],
        tdvf_finite_impedance.real[frequency_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="TD-VF finite-wake impedance",
    )
    impedance_axes[0].plot(
        cst_frequencies[frequency_mask],
        tdvf_fully_decayed_wake_potential_impedance.real[
            frequency_mask
        ],
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label="TD-VF analytical wake-potential impedance",
    )

    impedance_axes[1].plot(
        cst_frequencies[frequency_mask],
        cst_finite_impedance.imag[frequency_mask],
        color="black",
        linewidth=1.5,
        label="CST finite-wake impedance",
    )

    impedance_axes[1].plot(
        cst_frequencies[frequency_mask],
        tdvf_finite_impedance.imag[frequency_mask],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="TD-VF finite-wake impedance",
    )

    impedance_axes[1].plot(
        cst_frequencies[frequency_mask],
        tdvf_fully_decayed_wake_potential_impedance.imag[
            frequency_mask
        ],
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label="TD-VF analytical wake-potential impedance",
    )

    impedance_axes[2].semilogy(
        cst_frequencies[frequency_mask],
        np.maximum(
            np.abs(cst_finite_impedance[frequency_mask]),
            np.finfo(float).eps,
        ),
        color="black",
        linewidth=1.5,
        label="CST finite-wake impedance",
    )

    impedance_axes[2].semilogy(
        cst_frequencies[frequency_mask],
        np.maximum(
            np.abs(tdvf_finite_impedance[frequency_mask]),
            np.finfo(float).eps,
        ),
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="TD-VF finite-wake impedance",
    )

    impedance_axes[2].semilogy(
        cst_frequencies[frequency_mask],
        np.maximum(
            np.abs(
                tdvf_fully_decayed_wake_potential_impedance[
                    frequency_mask
                ]
            ),
            np.finfo(float).eps,
        ),
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        label="TD-VF analytical wake-potential impedance",
    )

    impedance_axes[0].set_ylabel(
        r"$\operatorname{Re}(Z_\perp)$"
    )
    impedance_axes[1].set_ylabel(
        r"$\operatorname{Im}(Z_\perp)$"
    )
    impedance_axes[2].set_ylabel(
        r"$|Z_\perp|$"
    )
    impedance_axes[2].set_xlabel("Frequency [Hz]")

    for axis in impedance_axes:
        axis.grid(True, which="both")
        axis.legend()

    plt.show()


if __name__ == "__main__":
    main()