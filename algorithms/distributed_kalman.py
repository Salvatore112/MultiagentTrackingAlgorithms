import numpy as np
from typing import Dict, List, Set, Optional, Any


class Distributed_Kalman:
    def __init__(
        self,
        sensors_positions: Optional[Dict[int, np.ndarray]] = None,
        true_targets_position: Optional[Dict[int, np.ndarray]] = None,
        distances: Optional[Dict[int, Dict[int, float]]] = None,
        init_coords: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    ) -> None:
        self.number_of_sensors: int = len(sensors_positions.items())
        self.sensor_ids: Set[int] = {i for i in range(self.number_of_sensors)}
        self.sensors_positions: Optional[Dict[int, np.ndarray]] = sensors_positions
        self.number_of_targets: int = len(true_targets_position.keys())
        self.target_ids: Set[int] = {i for i in range(self.number_of_targets)}
        self.true_targets_position: Optional[Dict[int, np.ndarray]] = (
            true_targets_position
        )
        self.distances: Optional[Dict[int, Dict[int, float]]] = distances
        self.init_coords: Optional[Dict[int, Dict[int, np.ndarray]]] = init_coords

        self.dimensions: int = 2
        self.process_noise_q: float = 0.01
        self.measurement_noise_r: float = 0.1
        self.consensus_weight: float = 0.5
        self.max_condition_number: float = 0

        while self.max_condition_number < 3:
            weight = self._generate_matrix(self.number_of_sensors)
            condition_number = self._condition_number(weight)
            if condition_number > self.max_condition_number:
                self.max_condition_number = condition_number
                self.weight: np.ndarray = weight

        self.state_estimates: Dict[int, Dict[int, np.ndarray]] = {}
        self.covariances: Dict[int, Dict[int, np.ndarray]] = {}
        self.initialize_filters()

    def _generate_matrix(self, n: int) -> np.ndarray:
        raw_mat = np.random.rand(n, n)
        weight = np.tril(raw_mat) + np.tril(raw_mat, -1).T
        weight = np.around(weight, 1)
        np.fill_diagonal(weight, 0)
        weight = -weight + np.diag([n - 1] * n)
        return weight

    def _condition_number(self, matrix: np.ndarray) -> float:
        eig = np.linalg.eig(matrix)[0]
        eig_list = sorted([abs(n) for n in eig if abs(n) > 0.00001])
        return eig_list[-1] / eig_list[0]

    def initialize_filters(self) -> None:
        for target_id in self.target_ids:
            self.state_estimates[target_id] = {}
            self.covariances[target_id] = {}
            for sensor_id in self.sensor_ids:
                self.state_estimates[target_id][sensor_id] = self.init_coords[target_id][
                    sensor_id
                ].copy()
                self.covariances[target_id][sensor_id] = np.eye(self.dimensions) * 10.0

    def _measurement_model(
        self, target_position: np.ndarray, sensor_position: np.ndarray
    ) -> float:
        return np.linalg.norm(target_position - sensor_position) ** 2

    def _jacobian_h(
        self, target_position: np.ndarray, sensor_position: np.ndarray
    ) -> np.ndarray:
        diff = target_position - sensor_position
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return np.zeros(self.dimensions)
        return 2 * diff

    def _get_neighbors(self, sensor_id: int) -> List[int]:
        neighbors_mat = (self.weight != 0).astype(int)
        np.fill_diagonal(neighbors_mat, 0)
        return [idx for idx, val in enumerate(neighbors_mat[sensor_id]) if val == 1]

    def run_main_algorithm(self) -> Dict[int, Dict[int, np.ndarray]]:
        f = np.eye(self.dimensions)
        q = np.eye(self.dimensions) * self.process_noise_q

        for target_id in self.target_ids:
            true_pos = self.true_targets_position[target_id]

            for sensor_id in self.sensor_ids:
                x_prev = self.state_estimates[target_id][sensor_id]
                p_prev = self.covariances[target_id][sensor_id]

                x_pred = f @ x_prev
                p_pred = f @ p_prev @ f.T + q

                sensor_pos = self.sensors_positions[sensor_id]
                z_measured = self.distances[target_id][sensor_id]

                h = self._jacobian_h(x_pred, sensor_pos)
                h = h.reshape(1, -1)

                s = h @ p_pred @ h.T + self.measurement_noise_r
                k = p_pred @ h.T / s

                z_pred = self._measurement_model(x_pred, sensor_pos)
                x_update = x_pred + k.flatten() * (z_measured - z_pred)
                p_update = (np.eye(self.dimensions) - k @ h) @ p_pred

                self.state_estimates[target_id][sensor_id] = x_update
                self.covariances[target_id][sensor_id] = p_update

            for sensor_id in self.sensor_ids:
                neighbors = self._get_neighbors(sensor_id)
                if not neighbors:
                    continue

                neighbor_states = [
                    self.state_estimates[target_id][neighbor] for neighbor in neighbors
                ]
                avg_state = np.mean(neighbor_states, axis=0)
                self.state_estimates[target_id][sensor_id] = (
                    self.consensus_weight * self.state_estimates[target_id][sensor_id]
                    + (1 - self.consensus_weight) * avg_state
                )

        return self.state_estimates

    def run_n_iterations(self, data: Dict[int, Any]) -> Dict[int, Any]:
        from collections import defaultdict

        result = defaultdict()
        for iteration in data.keys():
            self.true_targets_position = data[iteration][0]
            self.distances = data[iteration][1]
            new_estimates = self.run_main_algorithm()
            self.init_coords = new_estimates
            result[iteration] = [data[iteration][0], new_estimates]
        return result