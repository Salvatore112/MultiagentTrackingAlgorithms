import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpRequest, HttpResponse
from django.urls import reverse
from typing import Dict, List, Any, Optional, Tuple
from .simulation import Simulation
from algorithms.original_spsa import Original_SPSA

matplotlib.use("Agg")


def setup_view(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        duration: float = float(request.POST.get("duration", 50))
        num_sensors: int = int(request.POST.get("num_sensors", 3))
        num_linear_targets: int = int(request.POST.get("num_linear_targets", 2))
        num_random_targets: int = int(request.POST.get("num_random_targets", 2))
        algorithms: List[str] = request.POST.getlist("algorithms")
        num_runs: int = int(request.POST.get("num_runs", 1))

        noise_enabled: bool = request.POST.get("noise_enabled") == "on"
        noise_low: float = float(request.POST.get("noise_low", -0.1))
        noise_high: float = float(request.POST.get("noise_high", 0.1))

        simulation_params = {
            "duration": duration,
            "num_sensors": num_sensors,
            "num_linear_targets": num_linear_targets,
            "num_random_targets": num_random_targets,
            "algorithms": algorithms,
            "noise_enabled": noise_enabled,
            "noise_low": noise_low,
            "noise_high": noise_high,
            "num_runs": num_runs,
        }

        request.session["simulation_params"] = simulation_params

        return HttpResponseRedirect(reverse("results"))

    return render(request, "simulations/setup.html")


def results_view(request: HttpRequest) -> HttpResponse:
    params: Dict[str, Any] = request.session.get("simulation_params", {})

    if not params:
        return HttpResponseRedirect(reverse("setup"))

    duration: float = params.get("duration", 50)
    num_sensors: int = params.get("num_sensors", 3)
    num_linear_targets: int = params.get("num_linear_targets", 2)
    num_random_targets: int = params.get("num_random_targets", 2)
    algorithms: List[str] = params.get("algorithms", ["original_spsa"])
    noise_enabled: bool = params.get("noise_enabled", False)
    noise_low: float = params.get("noise_low", -0.1)
    noise_high: float = params.get("noise_high", 0.1)
    num_runs: int = params.get("num_runs", 1)

    noise_config: Optional[Dict[str, Any]] = None
    if noise_enabled:
        noise_config = {"type": "uniform", "low": noise_low, "high": noise_high}

    all_results: Dict[int, Dict[str, Any]] = {}
    all_simulations: Dict[int, Simulation] = {}
    all_initial_estimates: Dict[int, Dict[int, Dict[int, np.ndarray]]] = {}

    for run_id in range(num_runs):
        sim: Simulation = Simulation(
            duration=duration, time_step=1.0, noise_config=noise_config
        )

        for i in range(num_sensors):
            sim.add_uniform_sensor(i, area_size=50)

        target_id: int = 0
        for i in range(num_linear_targets):
            sim.add_linear_target(target_id, area_size=50)
            target_id += 1

        for i in range(num_random_targets):
            sim.add_random_walk_target(target_id, area_size=50)
            target_id += 1

        sim.run_simulation()
        spsa_input: Dict[str, Any] = sim.get_spsa_input_data()

        all_initial_estimates[run_id] = spsa_input["init_coords"]

        results: Dict[str, Any] = {}
        if "original_spsa" in algorithms:
            test_obj: Original_SPSA = Original_SPSA(
                sensors_positions=spsa_input["sensors_positions"],
                true_targets_position=spsa_input["data"][0][0],
                distances=spsa_input["data"][0][1],
                init_coords=spsa_input["init_coords"],
            )
            results["original_spsa"] = test_obj.run_n_iterations(
                data=spsa_input["data"]
            )

        if "accelerated_spsa" in algorithms:
            from algorithms.accelerated_spsa import Accelerated_SPSA

            acc_test_obj: Accelerated_SPSA = Accelerated_SPSA(
                sensors_positions=spsa_input["sensors_positions"],
                true_targets_position=spsa_input["data"][0][0],
                distances=spsa_input["data"][0][1],
                init_coords=spsa_input["init_coords"],
            )
            results["accelerated_spsa"] = acc_test_obj.run_n_iterations(
                data=spsa_input["data"]
            )

        all_results[run_id] = results
        all_simulations[run_id] = sim

    selected_run: Optional[int] = request.GET.get("run")
    selected_sensor: Optional[str] = request.GET.get("sensor")
    selected_target: Optional[str] = request.GET.get("target")

    if selected_run is None or selected_run == "":
        selected_run = 0
    else:
        selected_run = int(selected_run)

    selected_run = max(0, min(selected_run, num_runs - 1))

    plots_data: Dict[str, str] = {}
    aggregated_plots: Dict[str, str] = {}

    if num_runs == 1:
        plots_data = generate_plots(
            all_simulations[0],
            all_results[0],
            all_initial_estimates[0],
            selected_run,
            num_runs,
        )
    else:
        aggregated_plots = generate_aggregated_plots(
            all_simulations, all_results, all_initial_estimates, num_runs
        )
        if selected_run < num_runs:
            plots_data = generate_plots(
                all_simulations[selected_run],
                all_results[selected_run],
                all_initial_estimates[selected_run],
                selected_run,
                num_runs,
            )

    context: Dict[str, Any] = {
        "plots": plots_data,
        "aggregated_plots": aggregated_plots,
        "results": all_results.get(selected_run, {}),
        "sensors": list(range(num_sensors)),
        "targets": list(range(num_linear_targets + num_random_targets)),
        "algorithms": algorithms,
        "simulation_params": params,
        "num_runs": num_runs,
        "selected_run": selected_run,
        "run_range": range(num_runs),
    }

    if (selected_sensor is not None and selected_sensor != "") or (
        selected_target is not None and selected_target != ""
    ):
        sensor_int: Optional[int] = (
            int(selected_sensor) if selected_sensor and selected_sensor != "" else None
        )
        target_int: Optional[int] = (
            int(selected_target) if selected_target and selected_target != "" else None
        )
        individual_plots: Dict[str, str] = generate_individual_plots(
            all_simulations[selected_run],
            all_results[selected_run],
            all_initial_estimates[selected_run],
            sensor_int,
            target_int,
            selected_run,
        )
        context["individual_plots"] = individual_plots
        context["selected_sensor"] = sensor_int
        context["selected_target"] = target_int

    return render(request, "simulations/results.html", context)


def generate_plots(
    sim: Simulation,
    results: Dict[str, Any],
    init_coords: Dict[int, Dict[int, np.ndarray]],
    run_id: int,
    num_runs: int,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    plt.figure(figsize=(12, 8))

    colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

    for i, target in enumerate(sim.targets):
        positions: List[Any] = [
            sim.get_target_position(target, t)
            for t in sim.simulation_data["time_points"]
        ]
        x_vals: List[float] = [p[0] for p in positions]
        y_vals: List[float] = [p[1] for p in positions]
        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linewidth=3,
            label=f"Target {target.id} (True)",
            alpha=0.7,
        )

        plt.scatter(
            x_vals[0],
            y_vals[0],
            color=colors[i],
            s=180,
            marker="D",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )

        plt.scatter(
            x_vals[-1],
            y_vals[-1],
            color=colors[i],
            s=180,
            marker="X",
            edgecolors="black",
            linewidth=2,
            zorder=5,
        )

    line_styles = {"original_spsa": "-", "accelerated_spsa": ":"}
    for algorithm_name, algorithm_results in results.items():
        for target_id in algorithm_results[0][0].keys():
            target_estimates: List[np.ndarray] = []

            initial_est = init_coords[target_id][0]
            target_estimates.append(initial_est)

            for time_iter in algorithm_results.values():
                estimates_at_time: Dict[int, np.ndarray] = time_iter[1][target_id]
                avg_estimate: np.ndarray = np.mean(
                    list(estimates_at_time.values()), axis=0
                )
                target_estimates.append(avg_estimate)

            x_vals: List[float] = [est[0] for est in target_estimates]
            y_vals: List[float] = [est[1] for est in target_estimates]

            plt.plot(
                x_vals,
                y_vals,
                line_styles.get(algorithm_name, "--"),
                color=colors[target_id],
                linewidth=2,
                label=f"Target {target_id} ({algorithm_name})",
                alpha=0.8,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=colors[target_id],
                s=120,
                marker="s",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=colors[target_id],
                s=120,
                marker="o",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

    for i, sensor in enumerate(sim.sensors):
        plt.scatter(
            sensor.position[0],
            sensor.position[1],
            color="red",
            s=150,
            marker="^",
            label=f"Sensor {sensor.id}",
            edgecolors="black",
            zorder=5,
        )
        plt.annotate(
            f"S{i}",
            (sensor.position[0], sensor.position[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.scatter(
        [],
        [],
        c="white",
        s=180,
        marker="D",
        edgecolors="black",
        linewidth=2,
        label="Start (True)",
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=180,
        marker="X",
        edgecolors="black",
        linewidth=2,
        label="End (True)",
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=120,
        marker="s",
        edgecolors="black",
        linewidth=2,
        label="Start (Est.)",
    )
    plt.scatter(
        [],
        [],
        c="white",
        s=120,
        marker="o",
        edgecolors="black",
        linewidth=2,
        label="End (Est.)",
    )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    if num_runs > 1:
        plt.title(
            f"True Trajectories and Algorithm Estimates (Run {run_id + 1}/{num_runs})"
        )
    else:
        plt.title("True Trajectories and Algorithm Estimates")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer: io.BytesIO = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    line_styles = {"original_spsa": "-", "accelerated_spsa": ":"}
    for algorithm_name, algorithm_results in results.items():
        errors_over_time: Dict[int, List[float]] = {
            target_id: [] for target_id in algorithm_results[0][0].keys()
        }

        for target_id in algorithm_results[0][0].keys():
            initial_est = init_coords[target_id][0]
            true_pos = algorithm_results[0][0][target_id]
            initial_error = np.linalg.norm(initial_est - true_pos)
            errors_over_time[target_id].append(initial_error)

        for time_iter in algorithm_results.values():
            true_positions: Dict[int, np.ndarray] = time_iter[0]
            estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

            for target_id, true_pos in true_positions.items():
                sensor_estimates: Dict[int, np.ndarray] = estimates[target_id]
                errors: List[float] = []
                for sensor_est in sensor_estimates.values():
                    error: float = np.linalg.norm(sensor_est - true_pos)
                    errors.append(error)
                avg_error: float = np.mean(errors)
                errors_over_time[target_id].append(avg_error)

        for target_id, errors in errors_over_time.items():
            plt.plot(
                range(len(errors)),
                errors,
                color=colors[target_id],
                linestyle=line_styles.get(algorithm_name, "-"),
                label=f"Target {target_id} ({algorithm_name})",
                linewidth=2,
            )

    plt.xlabel("Iteration (including initial)")
    plt.ylabel("Average Error (All Sensors)")
    if num_runs > 1:
        plt.title(f"Convergence Error for Each Target (Run {run_id + 1}/{num_runs})")
    else:
        plt.title("Convergence Error for Each Target")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["convergence"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots


def generate_aggregated_plots(
    all_simulations: Dict[int, Simulation],
    all_results: Dict[int, Dict[str, Any]],
    all_initial_estimates: Dict[int, Dict[int, Dict[int, np.ndarray]]],
    num_runs: int,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    if num_runs <= 1:
        return plots

    aggregated_errors: Dict[str, List[List[float]]] = {}

    for algorithm_name in all_results[0].keys():
        aggregated_errors[algorithm_name] = []
        for run_id in range(num_runs):
            if run_id in all_results and algorithm_name in all_results[run_id]:
                algorithm_results = all_results[run_id][algorithm_name]
                errors_over_time: List[float] = []

                init_coords = all_initial_estimates[run_id]

                initial_iteration_errors: List[float] = []
                for target_id, true_pos in algorithm_results[0][0].items():
                    initial_est = init_coords[target_id][0]
                    error: float = np.linalg.norm(initial_est - true_pos)
                    initial_iteration_errors.append(error)
                initial_avg_error = (
                    np.mean(initial_iteration_errors)
                    if initial_iteration_errors
                    else 0.0
                )
                errors_over_time.append(initial_avg_error)

                for time_iter in algorithm_results.values():
                    true_positions: Dict[int, np.ndarray] = time_iter[0]
                    estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                    iteration_errors: List[float] = []
                    for target_id, true_pos in true_positions.items():
                        sensor_estimates: Dict[int, np.ndarray] = estimates[target_id]
                        for sensor_est in sensor_estimates.values():
                            error: float = np.linalg.norm(sensor_est - true_pos)
                            iteration_errors.append(error)

                    avg_error: float = (
                        np.mean(iteration_errors) if iteration_errors else 0.0
                    )
                    errors_over_time.append(avg_error)

                aggregated_errors[algorithm_name].append(errors_over_time)

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(aggregated_errors.keys())))

    for idx, (algorithm_name, all_run_errors) in enumerate(aggregated_errors.items()):
        if not all_run_errors:
            continue

        min_length = min(len(errors) for errors in all_run_errors)
        all_run_errors = [errors[:min_length] for errors in all_run_errors]

        mean_errors = np.mean(all_run_errors, axis=0)
        std_errors = np.std(all_run_errors, axis=0)

        iterations = range(len(mean_errors))

        plt.plot(
            iterations,
            mean_errors,
            color=colors[idx],
            label=algorithm_name,
            linewidth=2,
        )
        plt.fill_between(
            iterations,
            mean_errors - std_errors,
            mean_errors + std_errors,
            color=colors[idx],
            alpha=0.2,
        )

    plt.xlabel("Iteration (including initial)")
    plt.ylabel("Aggregated Error (Mean ± Std)")
    plt.title(f"Aggregated Error Convergence ({num_runs} Runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["aggregated"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots


def generate_individual_plots(
    sim: Simulation,
    results: Dict[str, Any],
    init_coords: Dict[int, Dict[int, np.ndarray]],
    sensor_id: Optional[int] = None,
    target_id: Optional[int] = None,
    run_id: int = 0,
) -> Dict[str, str]:
    plots: Dict[str, str] = {}

    plt.figure(figsize=(12, 8))

    if target_id is not None:
        colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.sensors)))

        target_obj = next((t for t in sim.targets if t.id == target_id), None)
        if target_obj:
            positions: List[Any] = [
                sim.get_target_position(target_obj, t)
                for t in sim.simulation_data["time_points"]
            ]
            x_vals: List[float] = [p[0] for p in positions]
            y_vals: List[float] = [p[1] for p in positions]
            plt.plot(
                x_vals,
                y_vals,
                color="black",
                linewidth=4,
                label=f"Target {target_id} (True)",
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color="black",
                s=200,
                marker="D",
                edgecolors="white",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color="black",
                s=200,
                marker="X",
                edgecolors="white",
                linewidth=2,
                zorder=5,
            )

        line_styles = {"original_spsa": "-", "accelerated_spsa": ":"}
        for algorithm_name, algorithm_results in results.items():
            if sensor_id is not None:
                sensor_estimates: List[np.ndarray] = []

                initial_est = init_coords[target_id][sensor_id]
                sensor_estimates.append(initial_est)

                for time_iter in algorithm_results.values():
                    estimates_at_time: Dict[int, np.ndarray] = time_iter[1][target_id]
                    if sensor_id in estimates_at_time:
                        sensor_estimates.append(estimates_at_time[sensor_id])

                if sensor_estimates:
                    x_vals: List[float] = [est[0] for est in sensor_estimates]
                    y_vals: List[float] = [est[1] for est in sensor_estimates]

                    plt.plot(
                        x_vals,
                        y_vals,
                        line_styles.get(algorithm_name, "--"),
                        color=colors[sensor_id % len(colors)],
                        linewidth=2,
                        label=f"Sensor {sensor_id} ({algorithm_name})",
                        alpha=0.8,
                    )

                    plt.scatter(
                        x_vals[0],
                        y_vals[0],
                        color=colors[sensor_id % len(colors)],
                        s=120,
                        marker="s",
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                    )

                    plt.scatter(
                        x_vals[-1],
                        y_vals[-1],
                        color=colors[sensor_id % len(colors)],
                        s=120,
                        marker="o",
                        edgecolors="black",
                        linewidth=2,
                        zorder=5,
                    )
            else:
                for sensor_idx in range(len(sim.sensors)):
                    sensor_estimates: List[np.ndarray] = []

                    initial_est = init_coords[target_id][sensor_idx]
                    sensor_estimates.append(initial_est)

                    for time_iter in algorithm_results.values():
                        estimates_at_time: Dict[int, np.ndarray] = time_iter[1][
                            target_id
                        ]
                        if sensor_idx in estimates_at_time:
                            sensor_estimates.append(estimates_at_time[sensor_idx])

                    if sensor_estimates:
                        x_vals: List[float] = [est[0] for est in sensor_estimates]
                        y_vals: List[float] = [est[1] for est in sensor_estimates]

                        plt.plot(
                            x_vals,
                            y_vals,
                            line_styles.get(algorithm_name, "--"),
                            color=colors[sensor_idx],
                            linewidth=2,
                            label=f"Sensor {sensor_idx} ({algorithm_name})",
                            alpha=0.8,
                        )

                        plt.scatter(
                            x_vals[0],
                            y_vals[0],
                            color=colors[sensor_idx],
                            s=120,
                            marker="s",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

                        plt.scatter(
                            x_vals[-1],
                            y_vals[-1],
                            color=colors[sensor_idx],
                            s=120,
                            marker="o",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

        for i, sensor in enumerate(sim.sensors):
            if sensor_id is None or sensor.id == sensor_id:
                plt.scatter(
                    sensor.position[0],
                    sensor.position[1],
                    color=colors[i],
                    s=150,
                    marker="^",
                    label=f"Sensor {sensor.id}",
                    edgecolors="black",
                    zorder=5,
                )
                plt.annotate(
                    f"S{sensor.id}",
                    (sensor.position[0], sensor.position[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                )

        title_suffix = ""
        if sensor_id is not None and target_id is not None:
            title_suffix = f" - Sensor {sensor_id} & Target {target_id}"
        elif target_id is not None:
            title_suffix = f" - Target {target_id}"

        title_suffix += f" (Run {run_id + 1})"

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(f"Trajectories{title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
    else:
        colors: np.ndarray = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

        for i, target in enumerate(sim.targets):
            positions: List[Any] = [
                sim.get_target_position(target, t)
                for t in sim.simulation_data["time_points"]
            ]
            x_vals: List[float] = [p[0] for p in positions]
            y_vals: List[float] = [p[1] for p in positions]
            plt.plot(
                x_vals,
                y_vals,
                color=colors[i],
                linewidth=3,
                label=f"Target {target.id} (True)",
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=colors[i],
                s=180,
                marker="D",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=colors[i],
                s=180,
                marker="X",
                edgecolors="black",
                linewidth=2,
                zorder=5,
            )

        line_styles = {"original_spsa": "-", "accelerated_spsa": ":"}
        for algorithm_name, algorithm_results in results.items():
            if sensor_id is not None:
                for target_idx in algorithm_results[0][0].keys():
                    sensor_estimates: List[np.ndarray] = []

                    initial_est = init_coords[target_idx][sensor_id]
                    sensor_estimates.append(initial_est)

                    for time_iter in algorithm_results.values():
                        estimates_at_time: Dict[int, np.ndarray] = time_iter[1][
                            target_idx
                        ]
                        if sensor_id in estimates_at_time:
                            sensor_estimates.append(estimates_at_time[sensor_id])

                    if sensor_estimates:
                        x_vals: List[float] = [est[0] for est in sensor_estimates]
                        y_vals: List[float] = [est[1] for est in sensor_estimates]

                        plt.plot(
                            x_vals,
                            y_vals,
                            line_styles.get(algorithm_name, "--"),
                            color=colors[target_idx],
                            linewidth=2,
                            label=f"Target {target_idx} (Sensor {sensor_id} {algorithm_name})",
                            alpha=0.8,
                        )

                        plt.scatter(
                            x_vals[0],
                            y_vals[0],
                            color=colors[target_idx],
                            s=120,
                            marker="s",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

                        plt.scatter(
                            x_vals[-1],
                            y_vals[-1],
                            color=colors[target_idx],
                            s=120,
                            marker="o",
                            edgecolors="black",
                            linewidth=2,
                            zorder=5,
                        )

        for i, sensor in enumerate(sim.sensors):
            if sensor_id is None or sensor.id == sensor_id:
                plt.scatter(
                    sensor.position[0],
                    sensor.position[1],
                    color="red",
                    s=150,
                    marker="^",
                    label=f"Sensor {sensor.id}",
                    edgecolors="black",
                    zorder=5,
                )
                plt.annotate(
                    f"S{sensor.id}",
                    (sensor.position[0], sensor.position[1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                )

        title_suffix = ""
        if sensor_id is not None:
            title_suffix = f" - Sensor {sensor_id}"

        title_suffix += f" (Run {run_id + 1})"

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title(f"Trajectories{title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    buffer: io.BytesIO = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["individual_trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    line_styles = {"original_spsa": "-", "accelerated_spsa": ":"}
    for algorithm_name, algorithm_results in results.items():
        if target_id is not None:
            if sensor_id is not None:
                errors: List[float] = []

                initial_est = init_coords[target_id][sensor_id]
                true_pos = algorithm_results[0][0][target_id]
                initial_error = np.linalg.norm(initial_est - true_pos)
                errors.append(initial_error)

                for time_iter in algorithm_results.values():
                    true_positions: Dict[int, np.ndarray] = time_iter[0]
                    estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                    if target_id in true_positions and target_id in estimates:
                        if sensor_id in estimates[target_id]:
                            true_pos: np.ndarray = true_positions[target_id]
                            sensor_est: np.ndarray = estimates[target_id][sensor_id]
                            error: float = np.linalg.norm(sensor_est - true_pos)
                            errors.append(error)

                if errors:
                    plt.plot(
                        range(len(errors)),
                        errors,
                        linestyle=line_styles.get(algorithm_name, "-"),
                        label=f"Sensor {sensor_id} ({algorithm_name})",
                        linewidth=2,
                    )
            else:
                colors_sensor: np.ndarray = plt.cm.tab10(
                    np.linspace(0, 1, len(sim.sensors))
                )
                for sensor_idx in range(len(sim.sensors)):
                    errors: List[float] = []

                    initial_est = init_coords[target_id][sensor_idx]
                    true_pos = algorithm_results[0][0][target_id]
                    initial_error = np.linalg.norm(initial_est - true_pos)
                    errors.append(initial_error)

                    for time_iter in algorithm_results.values():
                        true_positions: Dict[int, np.ndarray] = time_iter[0]
                        estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                        if target_id in true_positions and target_id in estimates:
                            if sensor_idx in estimates[target_id]:
                                true_pos: np.ndarray = true_positions[target_id]
                                sensor_est: np.ndarray = estimates[target_id][
                                    sensor_idx
                                ]
                                error: float = np.linalg.norm(sensor_est - true_pos)
                                errors.append(error)

                    if errors:
                        plt.plot(
                            range(len(errors)),
                            errors,
                            color=colors_sensor[sensor_idx],
                            linestyle=line_styles.get(algorithm_name, "-"),
                            label=f"Sensor {sensor_idx} ({algorithm_name})",
                            linewidth=2,
                        )
        else:
            if sensor_id is not None:
                colors_target: np.ndarray = plt.cm.tab10(
                    np.linspace(0, 1, len(sim.targets))
                )
                for target_idx in algorithm_results[0][0].keys():
                    errors: List[float] = []

                    initial_est = init_coords[target_idx][sensor_id]
                    true_pos = algorithm_results[0][0][target_idx]
                    initial_error = np.linalg.norm(initial_est - true_pos)
                    errors.append(initial_error)

                    for time_iter in algorithm_results.values():
                        true_positions: Dict[int, np.ndarray] = time_iter[0]
                        estimates: Dict[int, Dict[int, np.ndarray]] = time_iter[1]

                        if target_idx in true_positions and target_idx in estimates:
                            if sensor_id in estimates[target_idx]:
                                true_pos: np.ndarray = true_positions[target_idx]
                                sensor_est: np.ndarray = estimates[target_idx][
                                    sensor_id
                                ]
                                error: float = np.linalg.norm(sensor_est - true_pos)
                                errors.append(error)

                    if errors:
                        plt.plot(
                            range(len(errors)),
                            errors,
                            color=colors_target[target_idx],
                            linestyle=line_styles.get(algorithm_name, "-"),
                            label=f"Target {target_idx} ({algorithm_name})",
                            linewidth=2,
                        )

    title_suffix = ""
    if sensor_id is not None and target_id is not None:
        title_suffix = f" - Sensor {sensor_id} & Target {target_id}"
    elif sensor_id is not None:
        title_suffix = f" - Sensor {sensor_id}"
    elif target_id is not None:
        title_suffix = f" - Target {target_id}"

    title_suffix += f" (Run {run_id + 1})"

    plt.xlabel("Iteration (including initial)")
    plt.ylabel("Error")
    plt.title(f"Convergence Error{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["individual_convergence"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots
