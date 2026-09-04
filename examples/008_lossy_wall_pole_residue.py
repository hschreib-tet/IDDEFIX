"""Benchmark real-pole fitting of a longitudinal lossy-wall impedance."""

import matplotlib.pyplot as plt
import numpy as np

from iddefix.poleResidueFitting import fit_poles_evolutionary


def longitudinal_lossy_wall_impedance(
    frequencies,
    pipe_radius,
    pipe_length,
    conductivity,
):
    """Return the thick-wall impedance of a circular beam pipe."""
    mu_0 = 4.0e-7 * np.pi
    s = 2j * np.pi * frequencies

    coefficient = (
        pipe_length
        / (2.0 * np.pi * pipe_radius)
        * np.sqrt(mu_0 / conductivity)
    )

    return coefficient * np.sqrt(s)


def build_real_pole_bounds(
    number_poles,
    minimum_frequency,
    maximum_frequency,
):
    """Build non-overlapping logarithmic bounds for real poles."""
    minimum_rate = 2.0 * np.pi * minimum_frequency / 100.0
    maximum_rate = 2.0 * np.pi * maximum_frequency * 100.0

    edges = np.linspace(
        np.log10(minimum_rate),
        np.log10(maximum_rate),
        number_poles + 1,
    )

    return [
        (edges[index], edges[index + 1])
        for index in range(number_poles)
    ]


def main():
    pipe_radius = 20.0e-3
    pipe_length = 1.0
    conductivity = 5.8e7

    minimum_frequency = 1.0e4
    maximum_frequency = 1.0e9

    frequencies = np.logspace(
        np.log10(minimum_frequency),
        np.log10(maximum_frequency),
        300,
    )

    target_impedance = longitudinal_lossy_wall_impedance(
        frequencies=frequencies,
        pipe_radius=pipe_radius,
        pipe_length=pipe_length,
        conductivity=conductivity,
    )

    numbers_of_poles = [1, 2, 4, 6, 8]
    results = {}

    for number_poles in numbers_of_poles:
        print(
            f"Fitting lossy-wall impedance with "
            f"{number_poles} real poles..."
        )

        result = fit_poles_evolutionary(
            frequencies=frequencies,
            impedance=target_impedance,
            number_real_poles=number_poles,
            number_complex_pairs=0,
            parameter_bounds=build_real_pole_bounds(
                number_poles=number_poles,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            ),
            fit_direct_term=True,
            amplitude_weighting="relative",
            frequency_weighting="log",
            maxiter=300,
            popsize=12,
            tol=1.0e-7,
            polish=True,
            seed=1234 + number_poles,
            workers=1,
        )

        fitted_impedance = result.residue_fit.fitted_impedance

        relative_error = np.abs(
            (fitted_impedance - target_impedance)
            / target_impedance
        )

        rms_relative_error = np.sqrt(
            np.mean(relative_error**2)
        )

        maximum_relative_error = np.max(relative_error)

        print(f"  objective: {result.objective_value:.6e}")
        print(f"  RMS relative error: {rms_relative_error:.6e}")
        print(f"  max relative error: {maximum_relative_error:.6e}")
        print(f"  direct term: {result.residue_fit.direct_term:.6e}")
        print(f"  real poles: {result.real_poles}")
        print(f"  residues: {result.residue_fit.residues}")
        print()

        results[number_poles] = {
            "fit": fitted_impedance,
            "relative_error": relative_error,
            "rms_relative_error": rms_relative_error,
        }

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    axes[0].loglog(
        frequencies,
        target_impedance.real,
        color="black",
        linewidth=2.5,
        label="Lossy-wall target: real part",
    )

    axes[0].loglog(
        frequencies,
        target_impedance.imag,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Lossy-wall target: imaginary part",
    )

    for number_poles, result in results.items():
        fitted_impedance = result["fit"]

        line = axes[0].loglog(
            frequencies,
            fitted_impedance.real,
            label=f"Fit: {number_poles} real poles",
        )[0]

        axes[0].loglog(
            frequencies,
            fitted_impedance.imag,
            color=line.get_color(),
            linestyle="--",
        )

    axes[0].set_ylabel("Longitudinal impedance [Ohm]")
    axes[0].grid(True, which="both")
    axes[0].legend()

    for number_poles, result in results.items():
        axes[1].loglog(
            frequencies,
            result["relative_error"],
            label=f"{number_poles} real poles",
        )

    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Pointwise relative error")
    axes[1].grid(True, which="both")
    axes[1].legend()

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()