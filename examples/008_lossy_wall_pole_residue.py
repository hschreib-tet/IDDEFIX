"""Benchmark real-pole fitting of a longitudinal lossy-wall impedance."""

import matplotlib.pyplot as plt
import numpy as np

import iddefix

from iddefix.poleResidueFitting import (
    build_fit_weights,
    fit_poles_evolutionary,
)


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


def weighted_resonator_error(
    parameters,
    fitFunction,
    x,
    y,
):
    """Weighted complex error for the original resonator model."""
    predicted = fitFunction(x, parameters)

    weights = build_fit_weights(
        frequencies=x,
        impedance=y,
        amplitude_weighting="relative",
        frequency_weighting="log",
    )

    error = weights * (predicted - y)

    return float(np.sum(np.abs(error) ** 2))

def build_resonator_bounds(
    number_resonators,
    minimum_frequency,
    maximum_frequency,
    maximum_impedance,
):
    """Build R, Q and resonance-frequency bounds."""
    minimum_resonance_frequency = minimum_frequency / 100.0
    maximum_resonance_frequency = maximum_frequency * 100.0

    frequency_edges = np.logspace(
        np.log10(minimum_resonance_frequency),
        np.log10(maximum_resonance_frequency),
        number_resonators + 1,
    )

    bounds = []

    for index in range(number_resonators):
        bounds.extend(
            [
                # Shunt impedance
                (0.0, 10.0 * maximum_impedance),
                # Quality factor
                (1.0e-3, 0.5),
                # Resonant frequency
                (
                    frequency_edges[index],
                    frequency_edges[index + 1],
                ),
            ]
        )

    return bounds

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

        resonator_results = {}

    for number_resonators in [1, 2, 3]:

        np.random.seed(4321 + number_resonators)
        
        equivalent_number_poles = 2 * number_resonators

        print(
            f"Fitting lossy-wall impedance with "
            f"{number_resonators} resonators "
            f"({equivalent_number_poles} poles)..."
        )

        model = iddefix.EvolutionaryAlgorithm(
            x_data=frequencies,
            y_data=target_impedance,
            N_resonators=number_resonators,
            parameterBounds=build_resonator_bounds(
                number_resonators=number_resonators,
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
                maximum_impedance=np.max(
                    np.abs(target_impedance)
                ),
            ),
            plane="longitudinal",
            objectiveFunction=weighted_resonator_error,
        )

        model.run_differential_evolution(
            maxiter=1000,
            popsize=12,
            mutation=(0.1, 0.5),
            crossover_rate=0.8,
            tol=1.0e-7,
            solver="scipy",
        )
        model.run_minimization_algorithm()

        fitted_impedance = model.get_impedance(
            frequency_data=frequencies,
            use_minimization=True,
        )

        relative_error = np.abs(
            (fitted_impedance - target_impedance)
            / target_impedance
        )

        rms_relative_error = np.sqrt(
            np.mean(relative_error**2)
        )

        maximum_relative_error = np.max(relative_error)

        print(f"  RMS relative error: {rms_relative_error:.6e}")
        print(f"  max relative error: {maximum_relative_error:.6e}")
        print(
            f"  parameters: {model.minimizationParameters}"
        )
        print()

        resonator_results[equivalent_number_poles] = {
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
    comparison_figure, comparison_axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    comparison_axes[0].loglog(
        frequencies,
        target_impedance.real,
        color="black",
        linewidth=2.5,
        label="Lossy-wall target: real",
    )

    comparison_axes[0].loglog(
        frequencies,
        target_impedance.imag,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Lossy-wall target: imaginary",
    )

    for number_poles in [2, 4, 6]:
        pole_fit = results[number_poles]["fit"]
        resonator_fit = resonator_results[number_poles]["fit"]

        comparison_axes[0].loglog(
            frequencies,
            pole_fit.real,
            label=f"{number_poles} real poles",
        )

        comparison_axes[0].loglog(
            frequencies,
            resonator_fit.real,
            linestyle=":",
            label=f"{number_poles // 2} resonators",
        )

    comparison_axes[0].set_ylabel(
        "Longitudinal impedance [Ohm]"
    )
    comparison_axes[0].grid(True, which="both")
    comparison_axes[0].legend()

    for number_poles in [2, 4, 6]:
        comparison_axes[1].loglog(
            frequencies,
            results[number_poles]["relative_error"],
            label=f"{number_poles} real poles",
        )

        comparison_axes[1].loglog(
            frequencies,
            resonator_results[number_poles][
                "relative_error"
            ],
            linestyle=":",
            label=f"{number_poles // 2} resonators",
        )

    comparison_axes[1].set_xlabel("Frequency [Hz]")
    comparison_axes[1].set_ylabel(
        "Pointwise relative error"
    )
    comparison_axes[1].grid(True, which="both")
    comparison_axes[1].legend()

    comparison_figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()