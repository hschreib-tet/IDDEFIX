"""Benchmark for the initial Time-Domain Vector Fitting implementation.

A known rational system is excited by a Gaussian pulse. Its response is
generated independently using scipy.integrate.solve_ivp.

Only a truncated part of the response is passed to TD-VF. The fitted
model is subsequently evaluated over the complete time interval to test
its extrapolation capability.
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
    """Return the Gaussian excitation signal."""

    return np.exp(
        -0.5
        * ((times - center) / sigma) ** 2
    )


def generate_reference_response(
    times,
    poles,
    residues,
):
    """Generate an independent reference response using an ODE solver.

    Each internal state satisfies

        dz_n/dt = p_n z_n + x(t),

    and the output is

        y(t) = sum_n R_n z_n(t).
    """

    pulse_center = 0.6e-9
    pulse_sigma = 0.12e-9

    def state_equation(time, states):
        excitation = gaussian_pulse(
            time,
            center=pulse_center,
            sigma=pulse_sigma,
        )

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
        raise RuntimeError(
            "Reference integration failed: "
            + solution.message
        )

    output_signal = np.sum(
        residues[:, None] * solution.y,
        axis=0,
    )

    return np.real_if_close(output_signal).real


def match_poles(
    reference_poles,
    fitted_poles,
):
    """Match fitted poles to reference poles."""

    distances = np.abs(
        fitted_poles[:, None]
        - reference_poles[None, :]
    )

    fitted_indices, reference_indices = (
        linear_sum_assignment(distances)
    )

    ordered_poles = np.empty_like(reference_poles)
    ordered_poles[reference_indices] = fitted_poles[
        fitted_indices
    ]

    return ordered_poles


def evaluate_fitted_model(
    times,
    input_signal,
    poles,
    residues,
    direct_term,
):
    """Evaluate a fitted rational model over a complete time interval."""

    time_step = times[1] - times[0]

    output_signal = (
        direct_term * input_signal.astype(complex)
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

        output_signal += residue * filtered_input

    return output_signal


def normalized_l2_error(
    reference,
    approximation,
):
    """Calculate a normalized L2 error."""

    return (
        np.linalg.norm(approximation - reference)
        / np.linalg.norm(reference)
    )


def main():
    # Complete reference interval
    times = np.linspace(
        0.0,
        40.0e-9,
        8001,
    )

    input_signal = gaussian_pulse(times)

    true_poles = np.array(
        [
            -8.0e7
            + 1j * 2.0 * np.pi * 1.1e9,
            -8.0e7
            - 1j * 2.0 * np.pi * 1.1e9,
        ]
    )

    true_residues = np.array(
        [
            5.0e10 + 2.0e10j,
            5.0e10 - 2.0e10j,
        ]
    )

    reference_output = generate_reference_response(
        times=times,
        poles=true_poles,
        residues=true_residues,
    )

    # Deliberately inaccurate starting poles
    initial_poles = np.array(
        [
            -3.0e8
            + 1j * 2.0 * np.pi * 0.9e9,
            -3.0e8
            - 1j * 2.0 * np.pi * 0.9e9,
        ]
    )

    decay_time = 1.0 / abs(true_poles[0].real)

    observation_times = np.array(
        [
            3.0e-9,
            5.0e-9,
            10.0e-9,
        ]
    )

    results = []

    print()
    print("True poles:")
    for pole in true_poles:
        print(f"  {pole:.8e}")

    print()
    print(f"Decay time: {decay_time * 1.0e9:.3f} ns")

    for observation_time in observation_times:
        fit_mask = times <= observation_time

        result = time_domain_vector_fit(
            times=times[fit_mask],
            input_signal=input_signal[fit_mask],
            output_signal=reference_output[fit_mask],
            initial_poles=initial_poles,
            maximum_iterations=30,
            tolerance=1.0e-10,
        )

        fitted_poles = match_poles(
            true_poles,
            result.poles,
        )

        pole_errors = (
            np.abs(fitted_poles - true_poles)
            / np.abs(true_poles)
        )

        reconstructed_output = evaluate_fitted_model(
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

        imaginary_fraction = (
            np.linalg.norm(reconstructed_output.imag)
            / np.linalg.norm(reference_output)
        )

        results.append(
            {
                "observation_time": observation_time,
                "result": result,
                "fitted_poles": fitted_poles,
                "pole_error": np.max(pole_errors),
                "fit_error": fit_error,
                "extrapolation_error": extrapolation_error,
                "imaginary_fraction": imaginary_fraction,
                "reconstructed_output": reconstructed_real,
            }
        )

        print()
        print("-" * 72)
        print(
            "Observation time: "
            f"{observation_time * 1.0e9:.3f} ns"
        )
        print(
            "T / tau: "
            f"{observation_time / decay_time:.3f}"
        )
        print(
            f"Relocation iterations: {result.iterations}"
        )

        print("Fitted poles:")
        for pole in fitted_poles:
            print(f"  {pole:.8e}")

        print(
            "Maximum relative pole error: "
            f"{np.max(pole_errors):.6e}"
        )
        print(
            "Fit-window L2 error: "
            f"{fit_error:.6e}"
        )
        print(
            "Extrapolation L2 error: "
            f"{extrapolation_error:.6e}"
        )
        print(
            "Relative imaginary output: "
            f"{imaginary_fraction:.6e}"
        )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
        constrained_layout=True,
    )

    time_ns = times * 1.0e9

    axes[0].plot(
        time_ns,
        input_signal,
        color="black",
        linewidth=2.0,
    )

    axes[0].set_title("Gaussian excitation")
    axes[0].set_ylabel("x(t)")
    axes[0].grid(True)

    axes[1].plot(
        time_ns,
        reference_output,
        color="black",
        linewidth=2.0,
        label="Independent reference",
    )

    for benchmark in results:
        observation_time_ns = (
            benchmark["observation_time"] * 1.0e9
        )

        axes[1].plot(
            time_ns,
            benchmark["reconstructed_output"],
            linewidth=1.2,
            label=(
                "TD-VF, "
                f"T = {observation_time_ns:g} ns"
            ),
        )

        axes[1].axvline(
            observation_time_ns,
            color="gray",
            linestyle=":",
            linewidth=0.8,
        )

    axes[1].set_title(
        "TD-VF reconstruction and extrapolation"
    )
    axes[1].set_ylabel("y(t)")
    axes[1].grid(True)
    axes[1].legend()

    observation_ratios = np.array(
        [
            benchmark["observation_time"]
            / decay_time
            for benchmark in results
        ]
    )

    pole_errors = np.array(
        [
            benchmark["pole_error"]
            for benchmark in results
        ]
    )

    extrapolation_errors = np.array(
        [
            benchmark["extrapolation_error"]
            for benchmark in results
        ]
    )

    axes[2].semilogy(
        observation_ratios,
        pole_errors,
        marker="o",
        label="Maximum pole error",
    )

    axes[2].semilogy(
        observation_ratios,
        extrapolation_errors,
        marker="s",
        label="Extrapolation error",
    )

    axes[2].set_title(
        "Accuracy versus observation-window length"
    )
    axes[2].set_xlabel(r"Observation time $T/\tau$")
    axes[2].set_ylabel("Relative error")
    axes[2].grid(True, which="both")
    axes[2].legend()

    plt.show()


if __name__ == "__main__":
    main()