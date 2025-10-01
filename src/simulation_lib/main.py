import numpy as np
import matplotlib.colors as mcolors
import os
import random
import math
import matplotlib.pyplot as plt
import pickle
import hashlib

from enum import Enum
from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


class TARGET_TYPE(Enum):
    LINEAR = 1
    RANDOM_WALK = 2

    def __str__(self):
        if self == TARGET_TYPE.LINEAR:
            return "lin"
        elif self == TARGET_TYPE.RANDOM_WALK:
            return "ran_wlk"
        return self.name


@dataclass
class Target:
    id: int
    initial_position: Tuple[float, float]
    velocity: Tuple[float, float]
    movement_type: str = TARGET_TYPE.LINEAR
    random_walk_params: Optional[Dict] = None
    unique_hash: Optional[str] = None


@dataclass
class Sensor:
    id: int
    position: Tuple[float, float]


class Simulation:
    def __init__(
        self,
        duration: float,
        time_step: float = 1.0,
        output_dir: str = "simulation_results",
    ):
        self.duration = duration
        self.time_step = time_step
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self.simulation_data: Dict = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_sensor(self, sensor_id: int, position: Tuple[float, float]):
        self.sensors.append(Sensor(sensor_id, position))

    def add_target(
        self,
        obj_id: int,
        initial_position: Tuple[float, float],
        velocity: Tuple[float, float],
        movement_type: str = TARGET_TYPE.LINEAR,
        random_walk_params: Optional[Dict] = None,
    ):
        unique_hash = hashlib.sha256(
            f"{obj_id}_{initial_position}_{velocity}_{movement_type}_{random_walk_params}".encode()
        ).hexdigest()
        self.targets.append(
            Target(
                obj_id,
                initial_position,
                velocity,
                movement_type,
                random_walk_params,
                unique_hash,
            )
        )

    def add_linear_target(self, obj_id: int, area_size: float = 50):
        seed = hashlib.sha256(f"linear_{obj_id}".encode()).hexdigest()
        random_state = random.getstate()
        random.seed(seed)

        initial_x = random.uniform(-area_size, area_size)
        initial_y = random.uniform(-area_size, area_size)
        speed = random.uniform(0.5, 3.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        random.setstate(random_state)

        self.targets.append(
            Target(
                obj_id, (initial_x, initial_y), (vx, vy), TARGET_TYPE.LINEAR, None, seed
            )
        )
        return (initial_x, initial_y), (vx, vy)

    def add_random_walk_target(self, obj_id: int, area_size: float = 50):
        seed = hashlib.sha256(f"random_walk_{obj_id}".encode()).hexdigest()
        random_state = random.getstate()
        random.seed(seed)

        initial_x = random.uniform(-area_size, area_size)
        initial_y = random.uniform(-area_size, area_size)
        speed = random.uniform(0.5, 2.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)
        random_walk_params = {
            "speed_variation": random.uniform(0.1, 0.5),
            "direction_change_prob": random.uniform(0.1, 0.3),
            "max_direction_change": math.pi / 4,
        }

        random.setstate(random_state)

        self.targets.append(
            Target(
                obj_id,
                (initial_x, initial_y),
                (vx, vy),
                TARGET_TYPE.RANDOM_WALK,
                random_walk_params,
                seed,
            )
        )
        return (initial_x, initial_y), (vx, vy)

    def add_uniform_sensor(self, sensor_id: int, area_size: float = 50):
        seed = hashlib.sha256(f"sensor_{sensor_id}".encode()).hexdigest()
        random_state = random.getstate()
        random.seed(seed)

        pos_x = random.uniform(-area_size, area_size)
        pos_y = random.uniform(-area_size, area_size)

        random.setstate(random_state)

        self.sensors.append(Sensor(sensor_id, (pos_x, pos_y)))
        return (pos_x, pos_y)

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_target_position(self, obj: Target, time: float) -> Tuple[float, float]:
        if obj.movement_type == TARGET_TYPE.LINEAR:
            return self._get_linear_position(obj, time)
        elif obj.movement_type == TARGET_TYPE.RANDOM_WALK:
            return self._get_random_walk_position(obj, time)
        else:
            return self._get_linear_position(obj, time)

    def _get_linear_position(self, obj: Target, time: float) -> Tuple[float, float]:
        x = obj.initial_position[0] + obj.velocity[0] * time
        y = obj.initial_position[1] + obj.velocity[1] * time
        return (x, y)

    def _get_random_walk_position(
        self, obj: Target, time: float
    ) -> Tuple[float, float]:
        random_state = random.getstate()
        random.seed(obj.unique_hash)

        time_points = np.arange(0, time + self.time_step, self.time_step)
        x, y = obj.initial_position
        current_vx, current_vy = obj.velocity
        params = obj.random_walk_params or {
            "speed_variation": 0.3,
            "direction_change_prob": 0.2,
            "max_direction_change": math.pi / 4,
        }

        for t in time_points[1:]:
            if random.random() < params["direction_change_prob"]:
                angle_change = random.uniform(
                    -params["max_direction_change"], params["max_direction_change"]
                )
                current_speed = math.sqrt(current_vx**2 + current_vy**2)
                current_angle = math.atan2(current_vy, current_vx)
                new_angle = current_angle + angle_change

                speed_variation = random.uniform(
                    1 - params["speed_variation"], 1 + params["speed_variation"]
                )
                new_speed = current_speed * speed_variation

                current_vx = new_speed * math.cos(new_angle)
                current_vy = new_speed * math.sin(new_angle)

            x += current_vx * self.time_step
            y += current_vy * self.time_step

        random.setstate(random_state)

        return (x, y)

    def run_simulation(self):
        time_points = np.arange(0, self.duration + self.time_step, self.time_step)

        self.simulation_data = {
            "time_points": time_points,
            "sensors": {sensor.id: sensor for sensor in self.sensors},
            "targets": {target.id: target for target in self.targets},
        }

    def get_distance(self, sensor_id: int, target_id: int, time: float) -> float:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        sensor = self.simulation_data["sensors"][sensor_id]
        target_obj = self.simulation_data["targets"][target_id]

        target_pos = self.get_target_position(target_obj, time)
        distance = self.calculate_distance(sensor.position, target_pos)

        return distance

    def get_target_position_at_time(
        self, target_id: int, time: float
    ) -> Tuple[float, float]:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        target_obj = self.simulation_data["targets"][target_id]
        return self.get_target_position(target_obj, time)

    def print_distances_at_time(self, time: float):
        print(f"\n=== Distances at time t={time} seconds ===")
        for sensor in self.sensors:
            for obj in self.targets:
                distance = self.get_distance(sensor.id, obj.id, time)
                obj_pos = self.get_target_position_at_time(obj.id, time)
                print(f"Sensor {sensor.id} -> Target {obj.id}: {distance:.2f} units")
                print(
                    f"  Target {obj.id} position: ({obj_pos[0]:.1f}, {obj_pos[1]:.1f})"
                )

    def print_multiple_times(self, times: List[float]):
        for time in times:
            self.print_distances_at_time(time)

    def plot_trajectories(self, save_file: bool = True):
        plt.figure(figsize=(12, 10))

        obj_colors = list(mcolors.TABLEAU_COLORS.keys())
        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]

        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]

            positions = [
                self.get_target_position(obj, t)
                for t in self.simulation_data["time_points"]
            ]
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]

            plt.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=2,
                label=f"Target {obj.id} ({obj.movement_type})",
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
                label=f"Target {obj.id} start",
            )
            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=color,
                s=100,
                marker="s",
                edgecolors="black",
                zorder=5,
                label=f"Target {obj.id} finish",
            )

            if len(positions) > 1 and obj.movement_type == TARGET_TYPE.LINEAR:
                mid_idx = len(positions) // 2
                plt.annotate(
                    "",
                    xy=positions[mid_idx + 1],
                    xytext=positions[mid_idx],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )

        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            plt.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
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

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title("Target trajectories and sensor positions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        if save_file:
            filename = os.path.join(self.output_dir, "trajectories.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Trajectories saved to: {filename}")

        plt.show()

    def create_animation(self, interval: int = 100, save_gif: bool = True):
        fig, ax = plt.subplots(figsize=(12, 10))
        time_points = self.simulation_data["time_points"]

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title("Target movement animation")
        ax.grid(True, alpha=0.3)

        all_x = []
        all_y = []
        for obj in self.targets:
            positions = [self.get_target_position(obj, t) for t in time_points]
            all_x.extend([p[0] for p in positions])
            all_y.extend([p[1] for p in positions])
        for sensor in self.sensors:
            all_x.append(sensor.position[0])
            all_y.append(sensor.position[1])

        margin = 5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        obj_points = []
        obj_trails = []
        obj_colors = list(mcolors.TABLEAU_COLORS.keys())

        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]
            (trail,) = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
            (point,) = ax.plot(
                [], [], color=color, marker="o", markersize=10, markeredgecolor="black"
            )
            obj_trails.append(trail)
            obj_points.append(point)

            ax.text(
                obj.initial_position[0],
                obj.initial_position[1],
                f"O{obj.id}({str(obj.movement_type)})",
                fontsize=10,
                fontweight="bold",
            )

        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]
        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            ax.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=150,
                marker="^",
                label=f"Sensor {sensor.id}",
                edgecolors="black",
                zorder=5,
            )
            ax.annotate(
                f"S{sensor.id}",
                (sensor.position[0], sensor.position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
            )

        time_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        def animate(frame):
            current_time = time_points[frame]
            time_text.set_text(f"Time: {current_time:.1f} sec")

            for i, obj in enumerate(self.targets):
                trail_x = []
                trail_y = []
                # Compute all positions up to current frame on-demand
                for t in time_points[: frame + 1]:
                    pos = self.get_target_position(obj, t)
                    trail_x.append(pos[0])
                    trail_y.append(pos[1])

                obj_trails[i].set_data(trail_x, trail_y)

                # Compute current position on-demand
                current_pos = self.get_target_position(obj, current_time)
                obj_points[i].set_data([current_pos[0]], [current_pos[1]])

            return obj_trails + obj_points + [time_text]

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(time_points),
            interval=interval,
            blit=True,
            repeat=True,
        )

        plt.legend()
        plt.tight_layout()

        if save_gif:
            try:
                gif_filename = os.path.join(self.output_dir, "animation.gif")
                anim.save(
                    gif_filename, writer=PillowWriter(fps=1000 // interval), dpi=100
                )
                print(f"Animation saved to: {gif_filename}")
            except Exception as e:
                print(f"Error saving GIF: {e}")

        plt.show()

        return anim

    def save_simulation(self, filename: str = "simulation_data.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "duration": self.duration,
                    "time_step": self.time_step,
                    "sensors": self.sensors,
                    "targets": self.targets,
                    "simulation_data": self.simulation_data,
                },
                f,
            )
        print(f"Simulation saved to: {filepath}")

    def load_simulation(self, filename: str = "simulation_data.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.duration = data["duration"]
            self.time_step = data["time_step"]
            self.sensors = data["sensors"]
            self.targets = data["targets"]
            self.simulation_data = data["simulation_data"]
        print(f"Simulation loaded from: {filepath}")

    def print_simulation_info(self):
        print(f"Simulation duration: {self.duration} seconds")
        print(f"Time step: {self.time_step} seconds")
        print(f"Number of sensors: {len(self.sensors)}")
        print(f"Number of targets: {len(self.targets)}")
        print("\nSensors:")
        for sensor in self.sensors:
            print(
                f"  Sensor {sensor.id}: position ({sensor.position[0]:.1f}, {sensor.position[1]:.1f})"
            )
        print("\nTargets:")
        for obj in self.targets:
            movement_info = f", movement: {obj.movement_type}"
            if obj.movement_type == TARGET_TYPE.RANDOM_WALK and obj.random_walk_params:
                movement_info += f", params: {obj.random_walk_params}"
            print(
                f"  Target {obj.id}: initial position ({obj.initial_position[0]:.1f}, {obj.initial_position[1]:.1f}), "
                f"velocity ({obj.velocity[0]:.2f}, {obj.velocity[1]:.2f}){movement_info}"
            )


if __name__ == "__main__":
    sim = Simulation(duration=50, time_step=1.0, output_dir="simulation_results")

    print("Adding random sensors...")
    for i in range(1, 4):
        pos = sim.add_uniform_sensor(i, area_size=30)
        print(f"Sensor {i}: position ({pos[0]:.1f}, {pos[1]:.1f})")

    print("\nAdding linear targets...")
    for i in range(1, 4):
        init_pos, velocity = sim.add_linear_target(i, area_size=30)
        print(
            f"Linear Target {i}: initial position ({init_pos[0]:.1f}, {init_pos[1]:.1f}), "
            f"velocity ({velocity[0]:.2f}, {velocity[1]:.2f})"
        )

    print("\nAdding random walk targets...")
    for i in range(4, 6):
        init_pos, velocity = sim.add_random_walk_target(i, area_size=30)
        print(
            f"Random Walk Target {i}: initial position ({init_pos[0]:.1f}, {init_pos[1]:.1f}), "
            f"initial velocity ({velocity[0]:.2f}, {velocity[1]:.2f})"
        )

    sim.run_simulation()

    print("\n" + "=" * 50)
    sim.print_simulation_info()

    print("\nSaving simulation to file...")
    sim.save_simulation()

    print("\nCreating trajectory visualization...")
    sim.plot_trajectories()

    print("\nCreating animation...")
    sim.create_animation(interval=100, save_gif=True)

    print("\n" + "=" * 50)
    print("SPECIFIC DISTANCES AT SPECIFIC TIMES")
    print("=" * 50)

    sim.print_multiple_times([10, 25, 40])

    print("\n" + "=" * 50)
    print("INDIVIDUAL QUERIES")
    print("=" * 50)

    random_times = [random.randint(0, 50) for _ in range(3)]
    for time in random_times:
        distance1 = sim.get_distance(1, 1, time)
        distance2 = sim.get_distance(2, 4, time)
        print(
            f"Time t={time}: Sensor1->Target1(linear): {distance1:.2f}, Sensor2->Target4(random): {distance2:.2f}"
        )

    print("\n" + "=" * 50)
    print("DETAILED DISTANCE ANALYSIS - RANDOM WALK TARGETS")
    print("=" * 50)

    times_to_check = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    print("\nRandom Walk Target 4 distances over time:")
    print("Time | Sensor1 | Sensor2 | Sensor3 | Target Position")
    print("-" * 65)
    for time in times_to_check:
        dist1 = sim.get_distance(1, 4, time)
        dist2 = sim.get_distance(2, 4, time)
        dist3 = sim.get_distance(3, 4, time)
        pos = sim.get_target_position_at_time(4, time)
        print(
            f"{time:4.1f} | {dist1:7.2f} | {dist2:7.2f} | {dist3:7.2f} | ({pos[0]:6.1f}, {pos[1]:6.1f})"
        )

    print("\nRandom Walk Target 5 distances over time:")
    print("Time | Sensor1 | Sensor2 | Sensor3 | Target Position")
    print("-" * 65)
    for time in times_to_check:
        dist1 = sim.get_distance(1, 5, time)
        dist2 = sim.get_distance(2, 5, time)
        dist3 = sim.get_distance(3, 5, time)
        pos = sim.get_target_position_at_time(5, time)
        print(
            f"{time:4.1f} | {dist1:7.2f} | {dist2:7.2f} | {dist3:7.2f} | ({pos[0]:6.1f}, {pos[1]:6.1f})"
        )
