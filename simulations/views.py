import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from .simulation import Simulation
from algorithms.original_spsa import Original_SPSA
import json


def setup_view(request):
    if request.method == "POST":
        duration = float(request.POST.get("duration", 50))
        num_sensors = int(request.POST.get("num_sensors", 3))
        num_linear_targets = int(request.POST.get("num_linear_targets", 2))
        num_random_targets = int(request.POST.get("num_random_targets", 2))
        algorithms = request.POST.getlist("algorithms")

        request.session["simulation_params"] = {
            "duration": duration,
            "num_sensors": num_sensors,
            "num_linear_targets": num_linear_targets,
            "num_random_targets": num_random_targets,
            "algorithms": algorithms,
        }

        return HttpResponseRedirect(reverse("results"))

    return render(request, "simulations/setup.html")


def results_view(request):
    params = request.session.get("simulation_params", {})

    if not params:
        return HttpResponseRedirect(reverse("setup"))

    duration = params.get("duration", 50)
    num_sensors = params.get("num_sensors", 3)
    num_linear_targets = params.get("num_linear_targets", 2)
    num_random_targets = params.get("num_random_targets", 2)
    algorithms = params.get("algorithms", ["original_spsa"])

    sim = Simulation(duration=duration, time_step=1.0)

    for i in range(num_sensors):
        sim.add_uniform_sensor(i, area_size=50)

    target_id = 0
    for i in range(num_linear_targets):
        sim.add_linear_target(target_id, area_size=50)
        target_id += 1

    for i in range(num_random_targets):
        sim.add_random_walk_target(target_id, area_size=50)
        target_id += 1

    sim.run_simulation()
    spsa_input = sim.get_spsa_input_data()

    results = {}
    if "original_spsa" in algorithms:
        test_obj = Original_SPSA(
            sensors_positions=spsa_input["sensors_positions"],
            true_targets_position=spsa_input["data"][0][0],
            distances=spsa_input["data"][0][1],
            init_coords=spsa_input["init_coords"],
        )
        results["original_spsa"] = test_obj.run_n_iterations(data=spsa_input["data"])

    plots_data = generate_plots(sim, results, spsa_input)

    context = {
        "plots": plots_data,
        "results": results,
        "sensors": list(range(num_sensors)),
        "algorithms": algorithms,
        "simulation_params": params,
    }

    selected_sensor = request.GET.get("sensor")
    if selected_sensor is not None and selected_sensor != "":
        selected_sensor = int(selected_sensor)
        sensor_plots = generate_sensor_plots(sim, results, spsa_input, selected_sensor)
        context["sensor_plots"] = sensor_plots
        context["selected_sensor"] = selected_sensor

    return render(request, "simulations/results.html", context)


def generate_plots(sim, results, spsa_input):
    plots = {}

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

    for i, target in enumerate(sim.targets):
        positions = [
            sim.get_target_position(target, t)
            for t in sim.simulation_data["time_points"]
        ]
        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]
        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linewidth=3,
            label=f"Target {target.id} (True)",
            alpha=0.7,
        )

    for algorithm_name, algorithm_results in results.items():
        for target_id in algorithm_results[0][0].keys():
            target_estimates = []
            for time_iter in algorithm_results.values():
                estimates_at_time = time_iter[1][target_id]
                avg_estimate = np.mean(list(estimates_at_time.values()), axis=0)
                target_estimates.append(avg_estimate)

            x_vals = [est[0] for est in target_estimates]
            y_vals = [est[1] for est in target_estimates]
            plt.plot(
                x_vals,
                y_vals,
                "--",
                linewidth=2,
                label=f"Target {target_id} ({algorithm_name} Est.)",
                alpha=0.8,
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

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("True Trajectories and Algorithm Estimates")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    for algorithm_name, algorithm_results in results.items():
        errors_over_time = {
            target_id: [] for target_id in algorithm_results[0][0].keys()
        }

        for time_iter in algorithm_results.values():
            true_positions = time_iter[0]
            estimates = time_iter[1]

            for target_id, true_pos in true_positions.items():
                sensor_estimates = estimates[target_id]
                errors = []
                for sensor_est in sensor_estimates.values():
                    error = np.linalg.norm(sensor_est - true_pos)
                    errors.append(error)
                avg_error = np.mean(errors)
                errors_over_time[target_id].append(avg_error)

        for target_id, errors in errors_over_time.items():
            plt.plot(
                range(len(errors)),
                errors,
                label=f"Target {target_id} ({algorithm_name})",
                linewidth=2,
            )

    plt.xlabel("Iteration")
    plt.ylabel("Average Error (All Sensors)")
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


def generate_sensor_plots(sim, results, spsa_input, sensor_id):
    plots = {}

    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(sim.targets)))

    for i, target in enumerate(sim.targets):
        positions = [
            sim.get_target_position(target, t)
            for t in sim.simulation_data["time_points"]
        ]
        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]
        plt.plot(
            x_vals,
            y_vals,
            color=colors[i],
            linewidth=3,
            label=f"Target {target.id} (True)",
            alpha=0.7,
        )

    for algorithm_name, algorithm_results in results.items():
        for target_id in algorithm_results[0][0].keys():
            sensor_estimates = []
            for time_iter in algorithm_results.values():
                estimates_at_time = time_iter[1][target_id]
                if sensor_id in estimates_at_time:
                    sensor_estimates.append(estimates_at_time[sensor_id])

            if sensor_estimates:
                x_vals = [est[0] for est in sensor_estimates]
                y_vals = [est[1] for est in sensor_estimates]
                plt.plot(
                    x_vals,
                    y_vals,
                    "--",
                    linewidth=2,
                    label=f"Target {target_id} (Sensor {sensor_id} Est.)",
                    alpha=0.8,
                )

    sensor_pos = None
    for sensor in sim.sensors:
        if sensor.id == sensor_id:
            sensor_pos = sensor.position
            plt.scatter(
                sensor.position[0],
                sensor.position[1],
                color="red",
                s=200,
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
                fontsize=12,
            )

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(f"True Trajectories and Sensor {sensor_id} Estimates")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["sensor_trajectories"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(12, 8))

    for algorithm_name, algorithm_results in results.items():
        errors_over_time = {
            target_id: [] for target_id in algorithm_results[0][0].keys()
        }

        for time_iter in algorithm_results.values():
            true_positions = time_iter[0]
            estimates = time_iter[1]

            for target_id, true_pos in true_positions.items():
                if sensor_id in estimates[target_id]:
                    sensor_est = estimates[target_id][sensor_id]
                    error = np.linalg.norm(sensor_est - true_pos)
                    errors_over_time[target_id].append(error)

        for target_id, errors in errors_over_time.items():
            if errors:
                plt.plot(
                    range(len(errors)),
                    errors,
                    label=f"Target {target_id} (Sensor {sensor_id})",
                    linewidth=2,
                )

    plt.xlabel("Iteration")
    plt.ylabel(f"Error (Sensor {sensor_id})")
    plt.title(f"Convergence Error for Each Target - Sensor {sensor_id}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    plots["sensor_convergence"] = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return plots
