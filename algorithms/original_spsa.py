import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple

__version__ = "0.5.0"
print(f"Version: {__version__}")


class Parameters:
    def __init__(self):
        self.d = 2  # number of dimensions (2D space)
        self.n = 0  # will be set when sensors are added
        self.N: Set[int] = set()  # indexes of sensors
        self.s_est: Dict[int, np.ndarray] = {}  # estimated sensors coordinates
        self.s_init: Dict[
            int, np.ndarray
        ] = {}  # initial sensors coordinates (for fallback)
        self.s_norms: Dict[int, float] = {}  # norms of sensors coordinates
        self.meas: Dict[
            Tuple[int, int], float
        ] = {}  # distance measurements between sensors
        self.weight: np.ndarray = np.array([])  # weight matrix

        # Algorithm parameters
        self.beta_1 = 0.01
        self.beta_2 = 0.01
        self.alpha = 0.1
        self.gamma = 0.1
        self.b = 1.0

        # Validation parameters
        self.max_coord_value = 1e6  # Maximum reasonable coordinate value
        self.max_area_size = 20.0  # Maximum reasonable area size (20x20)
        self.default_distance = 10.0  # Default distance when no other option

    @property
    def beta(self):
        return self.beta_1 + self.beta_2

    def add_sensor(self, sensor_id: int):
        """Add a sensor with given ID"""
        if sensor_id not in self.N:
            self.N.add(sensor_id)
            # Initialize with random position
            initial_pos = np.array([np.random.rand() * 10, np.random.rand() * 10])
            self.s_init[sensor_id] = initial_pos.copy()
            self.s_est[sensor_id] = initial_pos.copy()
            self.s_norms[sensor_id] = np.sum(initial_pos * initial_pos)
            self.n = len(self.N)

    def add_measurement(self, from_sensor: int, to_sensor: int, distance: float):
        """Add a distance measurement between two sensors"""
        # Add sensors if they don't exist
        self.add_sensor(from_sensor)
        self.add_sensor(to_sensor)

        self.meas[(from_sensor, to_sensor)] = distance
        # Ensure symmetric measurements
        if (to_sensor, from_sensor) not in self.meas:
            self.meas[(to_sensor, from_sensor)] = distance

    def load_measurements_from_file(self, filename: str):
        """Load distance measurements from a text file"""
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    from_sensor = int(parts[0])
                    to_sensor = int(parts[1])
                    distance = float(parts[2])
                    self.add_measurement(from_sensor, to_sensor, distance)

    def generate_weight_matrix(self):
        """Generate weight matrix based on available measurements"""
        if self.n == 0:
            raise ValueError("No sensors defined")

        # Initialize weight matrix with zeros
        weight = np.zeros((self.n, self.n))

        # Create mapping from sensor IDs to matrix indices
        sensor_ids = sorted(self.N)
        id_to_idx = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}

        # Fill based on measurements
        for (i, j), dist in self.meas.items():
            row = id_to_idx[i]
            col = id_to_idx[j]
            weight[row, col] = 1  # connection exists
            weight[col, row] = 1  # symmetric

        # Ensure diagonal is zero
        np.fill_diagonal(weight, 0)

        # Normalize rows
        row_sums = weight.sum(axis=1)
        weight = weight / row_sums[:, np.newaxis]

        self.weight = weight
        return weight


class SPSA:
    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.d = parameters.d
        self.Delta_abs_value = 1 / np.sqrt(self.d)
        self.errors = {}

    def f_i(
        self, i: int, theta_i: np.ndarray, neighbors: Dict[int, List[int]]
    ) -> float:
        """Calculate function f for sensor i"""
        error = 0.0

        for j in neighbors.get(i, []):
            if (i, j) in self.params.meas:
                # Calculate squared distance between current estimate and measurement
                estimated_dist_sq = np.sum((theta_i - self.params.s_est[j]) ** 2)
                measured_dist_sq = self.params.meas[(i, j)] ** 2
                error += (estimated_dist_sq - measured_dist_sq) ** 2

        return error

    def get_neighbors(self, weight: np.ndarray) -> Dict[int, List[int]]:
        """Get neighbors for each sensor based on weight matrix"""
        # Create mapping from matrix indices to sensor IDs
        sensor_ids = sorted(self.params.N)
        idx_to_id = {idx: sensor_id for idx, sensor_id in enumerate(sensor_ids)}

        adjacency = (weight != 0).astype(int)
        np.fill_diagonal(adjacency, 0)

        neighbors = defaultdict(list)
        for idx in range(len(adjacency)):
            sensor_id = idx_to_id[idx]
            neighbors[sensor_id] = [
                idx_to_id[j] for j, connected in enumerate(adjacency[idx]) if connected
            ]

        return neighbors

    def validate_and_correct_coordinates(
        self, theta_hat: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Validate and correct unreasonable coordinates"""
        corrected = {}
        area_diagonal = np.sqrt(2) * self.params.max_area_size

        for sid, pos in theta_hat.items():
            # Check for unreasonable coordinates
            if (np.abs(pos) > self.params.max_coord_value).any():
                # Try to use initial position if reasonable
                if (
                    sid in self.params.s_init
                    and (
                        np.abs(self.params.s_init[sid]) <= self.params.max_area_size
                    ).all()
                ):
                    corrected[sid] = self.params.s_init[sid].copy()
                    print(
                        f"WARNING: Replaced unreasonable coordinates for sensor {sid} with initial position"
                    )
                else:
                    # Fallback to random position in 20x20 area
                    corrected[sid] = np.array(
                        [
                            np.random.rand() * self.params.max_area_size,
                            np.random.rand() * self.params.max_area_size,
                        ]
                    )
                    print(
                        f"WARNING: Replaced unreasonable coordinates for sensor {sid} with random position"
                    )
            else:
                corrected[sid] = pos.copy()

        return corrected

    def run(self, num_steps: int = 50, eps: float = 0.001):
        """Run the SPSA algorithm"""
        theta_hat = {sid: self.params.s_est[sid].copy() for sid in self.params.N}
        neighbors = self.get_neighbors(self.params.weight)

        for k in range(1, num_steps + 1):
            theta_new = {}
            err = 0.0

            for i in self.params.N:
                # Generate perturbation vector
                delta = (
                    np.array(
                        [1 if np.random.rand() < 0.5 else -1 for _ in range(self.d)]
                    )
                    * self.Delta_abs_value
                )

                # Evaluate at two perturbed points
                x1 = theta_hat[i] + self.params.beta_1 * delta
                x2 = theta_hat[i] - self.params.beta_2 * delta

                y1 = self.f_i(i, x1, neighbors)
                y2 = self.f_i(i, x2, neighbors)

                # SPSA gradient estimate
                spsa = (y1 - y2) / (2 * self.params.beta) * delta

                # Consensus term with neighbors
                consensus = np.zeros(self.d)
                for j in neighbors.get(i, []):
                    consensus += self.params.weight[i - min(self.params.N)][
                        j - min(self.params.N)
                    ] * (theta_hat[i] - theta_hat[j])

                # Update rule
                theta_new[i] = theta_hat[i] - (
                    self.params.alpha * spsa + self.params.gamma * consensus
                )

                # Calculate error contribution
                err += self.f_i(i, theta_new[i], neighbors)

            # Normalize error
            err /= len(self.params.N)

            theta_hat = theta_new.copy()
            self.errors[k] = err

            if err < eps or err > 1e9 or np.isnan(err):
                break

        # Validate and correct coordinates
        theta_hat = self.validate_and_correct_coordinates(theta_hat)

        # Save final positions to params
        for sid, pos in theta_hat.items():
            self.params.s_est[sid] = pos

        return theta_hat


def run_spsa_with_validation(
    params: Parameters, num_steps: int = 100
) -> Dict[int, np.ndarray]:
    """Run SPSA with additional validation steps"""
    spsa = SPSA(params)
    theta_hat = spsa.run(num_steps=num_steps)

    # Additional validation after running
    theta_hat = spsa.validate_and_correct_coordinates(theta_hat)

    return theta_hat


def calculate_safe_distance(params: Parameters, i: int, j: int) -> float:
    """Улучшенный расчет расстояний с использованием многоуровневой стратегии"""
    # 1. Проверка прямых измерений
    if (i, j) in params.meas:
        return params.meas[(i, j)]
    if (j, i) in params.meas:
        return params.meas[(j, i)]

    # 2. Проверка валидности координат
    pos_i = params.s_est.get(i)
    pos_j = params.s_est.get(j)

    if pos_i is not None and pos_j is not None:
        distance = np.linalg.norm(pos_i - pos_j)
        if distance <= params.max_area_size * 1.5:  # Небольшой запас
            return distance

    # 3. Поиск через общих соседей (2 шага)
    neighbor_dist = find_distance_for_neighbors(params, i, j, max_hops=2)
    if neighbor_dist is not None:
        return neighbor_dist

    # 4. Оценка на основе кластеризации
    cluster_dist = estimate_by_cluster(params, i, j)
    if cluster_dist is not None:
        return cluster_dist

    # 5. Финальный fallback - среднее из всех измерений
    if params.meas:
        avg_meas = np.mean(list(params.meas.values()))
        return min(avg_meas, params.max_area_size)

    return params.default_distance


def find_distance_for_neighbors(params, i, j, max_hops=2):
    """Поиск расстояния через соседей с ограничением на количество шагов"""
    from collections import deque

    # Поиск в ширину с ограничением глубины
    queue = deque([(i, 0, 0)])
    visited = {i: 0}
    paths = []

    while queue:
        current, dist, hops = queue.popleft()

        if hops > max_hops:
            continue

        # Получаем всех соседей текущего узла
        neighbors = set()
        for (a, b), d in params.meas.items():
            if a == current:
                neighbors.add((b, d))
            if b == current:
                neighbors.add((a, d))

        for neighbor, d in neighbors:
            if neighbor == j:
                paths.append(dist + d)
                continue

            if neighbor not in visited or visited[neighbor] > dist + d:
                visited[neighbor] = dist + d
                queue.append((neighbor, dist + d, hops + 1))

    if paths:
        return min(paths)
    return None


def estimate_by_cluster(params, i, j):
    """Оценка расстояния на основе кластеризации сенсоров"""
    # Получаем все известные позиции
    known_positions = {
        k: v
        for k, v in params.s_est.items()
        if v is not None and not np.any(np.abs(v) > params.max_coord_value)
    }

    if not known_positions:
        return None

    # Вычисляем среднюю плотность сенсоров
    all_distances = []
    for (a, b), d in params.meas.items():
        if a in known_positions and b in known_positions:
            all_distances.append(d)

    if all_distances:
        avg_distance = np.mean(all_distances)
        return min(avg_distance * 1.5, params.max_area_size)

    return None


def calculate_fallback_distance(params: Parameters, i: int, j: int) -> float:
    """Calculate fallback distance when direct calculation is impossible"""
    # Try to find any measured distance to these sensors
    i_distances = [d for (a, b), d in params.meas.items() if a == i or b == i]
    j_distances = [d for (a, b), d in params.meas.items() if a == j or b == j]

    # If we have some measurements, return average
    if i_distances and j_distances:
        avg_dist = (np.mean(i_distances) + np.mean(j_distances)) / 2
        return min(avg_dist, params.max_area_size)

    # Final fallback
    return params.default_distance


def calculate_distance_for_neighbors(
    params: Parameters, i: int, j: int, max_dist: float
) -> float:
    """Calculate distance by finding path through common neighbors"""
    # Find common neighbors
    neighbors_i = {b for (a, b) in params.meas.keys() if a == i} | {
        a for (a, b) in params.meas.keys() if b == i
    }
    neighbors_j = {b for (a, b) in params.meas.keys() if a == j} | {
        a for (a, b) in params.meas.keys() if b == j
    }
    common_neighbors = neighbors_i & neighbors_j

    if not common_neighbors:
        return min(params.default_distance, max_dist)

    # Calculate distance for each common neighbor
    distances = []
    for n in common_neighbors:
        try:
            d_in = params.meas.get((i, n), params.meas.get((n, i)))
            d_nj = params.meas.get((n, j), params.meas.get((j, n)))
            if d_in is not None and d_nj is not None:
                distances.append(d_in + d_nj)
        except KeyError:
            continue

    if distances:
        avg_distance = np.mean(distances)
        return min(avg_distance, max_dist)

    return min(params.default_distance, max_dist)


def print_distances(params: Parameters):
    """Улучшенный вывод расстояний"""
    print("\nAll distances between sensors:")
    print("From | To | Distance (Measured/Calculated)")
    print("-" * 45)

    sensor_ids = sorted(params.N)
    distance_cache = {}  # Кэш для хранения рассчитанных расстояний

    for i in sensor_ids:
        for j in sensor_ids:
            if i >= j:
                continue

            cache_key = (min(i, j), max(i, j))
            if cache_key in distance_cache:
                dist, source = distance_cache[cache_key]
            else:
                if (i, j) in params.meas or (j, i) in params.meas:
                    dist = params.meas.get((i, j), params.meas.get((j, i)))
                    source = "Measured"
                else:
                    dist = calculate_safe_distance(params, i, j)
                    source = "Calculated"
                distance_cache[cache_key] = (dist, source)

            print(f"{i:4d} | {j:2d} | {dist:10.4f} ({source:9})")


def plot_network_graph(params: Parameters):
    """Plot the network graph with distances and sensor IDs"""
    plt.figure(figsize=(15, 12))

    # Draw sensors
    sensor_ids = sorted(params.s_est.keys())
    x = [params.s_est[sid][0] for sid in sensor_ids]
    y = [params.s_est[sid][1] for sid in sensor_ids]

    plt.scatter(x, y, s=200, c="lightblue", edgecolors="blue", alpha=0.7)

    # Annotate sensor IDs
    for sid, x_pos, y_pos in zip(sensor_ids, x, y):
        plt.text(
            x_pos,
            y_pos,
            str(sid),
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Draw connections and distances
    drawn_pairs = set()
    for (i, j), dist in params.meas.items():
        if i < j:  # Draw each connection only once
            plt.plot(
                [params.s_est[i][0], params.s_est[j][0]],
                [params.s_est[i][1], params.s_est[j][1]],
                "gray",
                alpha=0.5,
                linewidth=0.5,
            )

            # Annotate distance at midpoint
            mid_x = (params.s_est[i][0] + params.s_est[j][0]) / 2
            mid_y = (params.s_est[i][1] + params.s_est[j][1]) / 2

            # Offset the text to avoid overlap
            offset_x = (params.s_est[j][1] - params.s_est[i][1]) * 0.1
            offset_y = (params.s_est[i][0] - params.s_est[j][0]) * 0.1

            plt.text(
                mid_x + offset_x,
                mid_y + offset_y,
                f"{dist:.2f}",
                fontsize=7,
                color="red",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

    plt.title("Network Graph with Estimated Positions and Distances", fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    # Initialize parameters
    params = Parameters()

    # Load measurements from file (will automatically create sensors)
    params.load_measurements_from_file("test_data.txt")

    # Generate weight matrix based on measurements
    params.generate_weight_matrix()

    # Create and run SPSA with validation
    theta_hat = run_spsa_with_validation(params, num_steps=100)

    # Plot the final network graph
    plot_network_graph(params)

    # Print all distances with validation
    print_distances(params)


if __name__ == "__main__":
    main()
