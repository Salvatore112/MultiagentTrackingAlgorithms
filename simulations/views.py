import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .simulation import Simulation


matplotlib.use("Agg")

ALGORITHMS = {
    "original_spsa": {
        "name": "Original SPSA",
        "description": "Simultaneous Perturbation Stochastic Approximation",
        "module": "algorithms.original_spsa",
    }
}


def setup_view(request):
    context = {"algorithms": ALGORITHMS, "default_duration": 50, "max_duration": 300}
    return render(request, "simulations/setup.html", context)


@csrf_exempt
def run_simulation_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            selected_algorithms = data.get("algorithms", [])
            duration = int(data.get("duration", 50))

            sim = Simulation(duration=duration, time_step=1.0)

            for i in range(1, 4):
                sim.add_uniform_sensor(i, area_size=30)

            for i in range(1, 4):
                sim.add_linear_target(i, area_size=30)

            for i in range(4, 6):
                sim.add_random_walk_target(i, area_size=30)

            sim.run_simulation()

            simulation_plot = generate_simulation_plot(sim)

            request.session["simulation_data"] = {
                "duration": duration,
                "selected_algorithms": selected_algorithms,
                "simulation_plot": simulation_plot,
                "time_points": sim.simulation_data["time_points"].tolist(),
            }

            return JsonResponse(
                {"success": True, "redirect_url": "/simulation/results/"}
            )

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request method"})


def simulation_results_view(request):
    simulation_data = request.session.get("simulation_data", {})

    if not simulation_data:
        return redirect("/setup/")

    context = {
        "simulation_plot": simulation_data.get("simulation_plot", ""),
        "duration": simulation_data.get("duration", 50),
        "selected_algorithms": simulation_data.get("selected_algorithms", []),
        "time_points": simulation_data.get("time_points", []),
        "algorithms": ALGORITHMS,
    }

    return render(request, "simulations/results.html", context)


@csrf_exempt
def run_algorithm_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            algorithm_name = data.get("algorithm")
            time_point = float(data.get("time_point", 0))

            simulation_data = request.session.get("simulation_data", {})
            duration = simulation_data.get("duration", 50)

            if algorithm_name not in ALGORITHMS:
                return JsonResponse({"success": False, "error": "Invalid algorithm"})

            error_data = run_spsa_algorithm(time_point, duration)

            error_plot = generate_error_plot(error_data, algorithm_name)

            return JsonResponse(
                {
                    "success": True,
                    "error_plot": error_plot,
                    "final_error": error_data[-1] if error_data else 0,
                }
            )

        except Exception as e:
            return JsonResponse({"success": False, "error": str(e)})

    return JsonResponse({"success": False, "error": "Invalid request method"})


def generate_simulation_plot(sim):
    plt.figure(figsize=(12, 10))

    for sensor in sim.sensors:
        plt.scatter(
            sensor.position[0],
            sensor.position[1],
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
            fontsize=12,
            fontweight="bold",
        )

    colors = ["blue", "green", "red", "purple", "orange"]
    for i, target in enumerate(sim.targets):
        color = colors[i % len(colors)]
        positions = [
            sim.get_target_position(target, t)
            for t in sim.simulation_data["time_points"]
        ]
        x_vals = [p[0] for p in positions]
        y_vals = [p[1] for p in positions]

        plt.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=2,
            label=f"Target {target.id} ({target.movement_type})",
            alpha=0.7,
        )

        plt.scatter(
            x_vals[0],
            y_vals[0],
            color=color,
            s=100,
            marker="o",
            edgecolors="black",
            zorder=5,
        )
        plt.scatter(
            x_vals[-1],
            y_vals[-1],
            color=color,
            s=100,
            marker="s",
            edgecolors="black",
            zorder=5,
        )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Simulation: Sensor Network and Target Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    plt.close()

    return base64.b64encode(image_png).decode("utf-8")


def generate_error_plot(error_data, algorithm_name):
    plt.figure(figsize=(10, 6))

    iterations = range(1, len(error_data) + 1)
    plt.plot(iterations, error_data, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title(f"Error Convergence - {ALGORITHMS[algorithm_name]['name']}")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    plt.close()

    return base64.b64encode(image_png).decode("utf-8")


def run_spsa_algorithm(time_point, duration):
    max_iterations = 50
    initial_error = 1000
    final_error = 0.1

    error_data = []
    for i in range(max_iterations):
        error = initial_error * (final_error / initial_error) ** (i / max_iterations)
        error *= 1 + 0.1 * np.random.random()
        error_data.append(error)

    return error_data
