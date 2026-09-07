"""TD-VF benchmark with real and complex-conjugate poles.

The reference system contains:

- one negative real pole,
- one strongly damped complex-conjugate pole pair,
- one weakly damped complex-conjugate pole pair.

The response is generated independently using solve_ivp and truncated
before the slowest mode has substantially decayed.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from iddefix.timeDomainVectorFitting import (
    recursive_exponential_convolution,
    time_domain_vector_fit,
)


def gaussian_pulse(
    times,
    center=0.6e-9,
    sigma=0.12e-9,
):
    """Gaussian excitation pulse."""

    return np.exp(
        -0.5
        * ((times - center) / sigma) ** 2
    )


def generate_reference_response(
    times,
    poles,
    residues,
):
    """Generate the reference response independently with solve_ivp."""

    def state_equation(time, states):
        excitation = gaussian_pulse(time)
        return poles * states + excitation

    solution = solve_ivp(
        fun=state_equation,
        t_span=(times[0], times[-1]),
        y0=np.zeros(poles.size, dtype=complex),
        t_eval=times,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=times[1] - times[0],
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    output_signal = np.sum(
        residues[:, None] * solution.y,
        axis=0,
    )

    return np.real_if_close(output_signal).real


def evaluate_model(
    times,
    input_signal,
    poles,
    residues,
    direct_term,
):
    """Evaluate the fitted rational model in the time domain."""

    time_step = times[1] - times[0]

    output_signal = (
        direct_term * input_signal.astype(complex)
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


def match_poles(reference_poles, fitted_poles):
    """Associate fitted poles with their closest reference poles."""

    distance_matrix = np.abs(
        fitted_poles[:, None]
        - reference_poles[None, :]
    )

    fitted_indices, reference_indices = (
        linear_sum_assignment(distance_matrix)
    )

    ordered_poles = np.empty_like(reference_poles)
    ordered_poles[reference_indices] = fitted_poles[
        fitted_indices
    ]

    return ordered_poles


def normalized_l2_error(reference, approximation):
    """Normalized L2 error."""

    return (
        np.linalg.norm(approximation - reference)
        / np.linalg.norm(reference)
    )


def pole_description(pole):
    """Classify a pole for the printed output."""

    imaginary_tolerance = (
        1.0e-10 * max(abs(pole), 1.0)
    )

    if abs(pole.imag) <= imaginary_tolerance:
        return "real"

    if pole.imag > 0.0:
        return "complex +"

    return "complex -"


def main():
    times = np.linspace(
        0.0,
        50.0e-9,
        10001,
    )

    input_signal = gaussian_pulse(times)

    # One real pole
    real_pole = -4.0e8

    # Strongly damped broadband resonance
    broadband_pole = (
        -6.0e8
        + 1j * 2.0 * np.pi * 0.70e9
    )

    # Weak, slowly decaying resonance
    narrowband_pole = (
        -5.0e7
        + 1j * 2.0 * np.pi * 1.15e9
    )

    true_poles = np.array(
        [
            real_pole,
            broadband_pole,
            np.conj(broadband_pole),
            narrowband_pole,
            np.conj(narrowband_pole),
        ],
        dtype=complex,
    )

    true_residues = np.array(
        [
            2.0e10,
            6.0e10 + 1.5e10j,
            6.0e10 - 1.5e10j,
            1.2e10 + 3.0e9j,
            1.2e10 - 3.0e9j,
        ],
        dtype=complex,
    )

    reference_output = generate_reference_response(
        times=times,
        poles=true_poles,
        residues=true_residues,
    )

    # Only the first 8 ns are made available to TD-VF.
    observation_time = 8.0e-9
    fit_mask = times <= observation_time

    # Deliberately imperfect initial poles.
    initial_broadband_pole = (
        -2.5e8
        + 1j * 2.0 * np.pi * 0.55e9
    )

    initial_narrowband_pole = (
        -2.0e8
        + 1j * 2.0 * np.pi * 1.35e9
    )

    initial_poles = np.array(
        [
            -1.0e9,
            initial_broadband_pole,
            np.conj(initial_broadband_pole),
            initial_narrowband_pole,
            np.conj(initial_narrowband_pole),
        ],
        dtype=complex,
    )

    result = time_domain_vector_fit(
        times=times[fit_mask],
        input_signal=input_signal[fit_mask],
        output_signal=reference_output[fit_mask],
        initial_poles=initial_poles,
        maximum_iterations=50,
        tolerance=1.0e-10,
    )

    fitted_poles = match_poles(
        true_poles,
        result.poles,
    )

    pole_errors = (
        np.abs(fitted_poles - true_poles)
        / np.maximum(
            np.abs(true_poles),
            np.finfo(float).eps,
        )
    )

    reconstructed_output = evaluate_model(
        times=times,
        input_signal=input_signal,
        poles=result.poles,
        residues=result.residues,
        direct_term=result.direct_term,
    )

    reconstructed_real = reconstructed_output.real

    fit_error = normalized_l2_error(
        reference_output[fit_mask],
        reconstructed_real[fit_mask],
    )

    extrapolation_mask = times > observation_time

    extrapolation_error = normalized_l2_error(
        reference_output[extrapolation_mask],
        reconstructed_real[extrapolation_mask],
    )

    imaginary_output_fraction = (
        np.linalg.norm(reconstructed_output.imag)
        / np.linalg.norm(reference_output)
    )

    print()
    print("TD-VF benchmark with mixed pole types")
    print("=" * 92)
    print(
        f"Observation time: {observation_time * 1.0e9:.2f} ns"
    )
    print(
        "Slowest decay time: "
        f"{1.0 / 5.0e7 * 1.0e9:.2f} ns"
    )
    print(
        "Observation time / slowest decay time: "
        f"{observation_time * 5.0e7:.3f}"
    )
    print(f"Relocation iterations: {result.iterations}")
    print()

    print(
        f"{'Type':<12}"
        f"{'Reference pole':>29}"
        f"{'Fitted pole':>29}"
        f"{'Relative error':>20}"
    )
    print("-" * 92)

    for true_pole, fitted_pole, error in zip(
        true_poles,
        fitted_poles,
        pole_errors,
    ):
        print(
            f"{pole_description(true_pole):<12}"
            f"{true_pole:>29.8e}"
            f"{fitted_pole:>29.8e}"
            f"{error:>20.6e}"
        )

    print("-" * 92)
    print(f"Fit-window L2 error:       {fit_error:.6e}")
    print(
        "Extrapolation L2 error:    "
        f"{extrapolation_error:.6e}"
    )
    print(
        "Relative imaginary output: "
        f"{imaginary_output_fraction:.6e}"
    )
    print(f"Direct term:               {result.direct_term:.8e}")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 12),
        constrained_layout=True,
    )

    time_ns = times * 1.0e9

    axes[0].plot(
        time_ns,
        reference_output,
        color="black",
        linewidth=2.0,
        label="Independent reference",
    )

    axes[0].plot(
        time_ns,
        reconstructed_real,
        color="tab:orange",
        linewidth=1.2,
        label="TD-VF reconstruction",
    )

    axes[0].axvspan(
        0.0,
        observation_time * 1.0e9,
        color="tab:blue",
        alpha=0.08,
        label="Data available to TD-VF",
    )

    axes[0].axvline(
        observation_time * 1.0e9,
        color="tab:red",
        linestyle="--",
        label="End of observation",
    )

    axes[0].set_title(
        "Reconstruction beyond the observation window"
    )
    axes[0].set_xlabel("Time [ns]")
    axes[0].set_ylabel("y(t)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        time_ns[fit_mask],
        reference_output[fit_mask],
        color="black",
        linewidth=2.0,
        label="Independent reference",
    )

    axes[1].plot(
        time_ns[fit_mask],
        reconstructed_real[fit_mask],
        color="tab:orange",
        linestyle="--",
        label="TD-VF fit",
    )

    axes[1].set_title("Response inside the fit window")
    axes[1].set_xlabel("Time [ns]")
    axes[1].set_ylabel("y(t)")
    axes[1].grid(True)
    axes[1].legend()

    # Pole plot:
    # horizontal coordinate = decay rate
    # vertical coordinate = oscillation frequency
    axes[2].scatter(
        -true_poles.real / 1.0e9,
        true_poles.imag / (2.0 * np.pi * 1.0e9),
        marker="x",
        s=100,
        linewidth=2.5,
        color="black",
        label="True poles",
    )

    axes[2].scatter(
        -initial_poles.real / 1.0e9,
        initial_poles.imag / (2.0 * np.pi * 1.0e9),
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="tab:blue",
        label="Initial poles",
    )

    axes[2].scatter(
        -result.poles.real / 1.0e9,
        result.poles.imag / (2.0 * np.pi * 1.0e9),
        marker="+",
        s=140,
        linewidth=2.5,
        color="tab:orange",
        label="Fitted poles",
    )

    axes[2].set_title("Initial, true and fitted pole locations")
    axes[2].set_xlabel(
        r"Damping rate $-\operatorname{Re}(p)$ [$10^9$ s$^{-1}$]"
    )
    axes[2].set_ylabel(
        r"Frequency $\operatorname{Im}(p)/(2\pi)$ [GHz]"
    )
    axes[2].grid(True)
    axes[2].legend()

    plt.show()


if __name__ == "__main__":
    main()