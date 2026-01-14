import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple


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
        self.process_noise_q: float = 0.1
        self.measurement_noise_r: float = 0.5
        self.consensus_iterations: int = 2
        self.consensus_weight: float = 0.3
        
        self.state_dim: int = 4
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.Q = np.eye(self.state_dim) * self.process_noise_q
        self.R = np.eye(2) * self.measurement_noise_r

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
                init_pos = self.init_coords[target_id][sensor_id]
                init_state = np.array([init_pos[0], init_pos[1], 0.0, 0.0])
                self.state_estimates[target_id][sensor_id] = init_state
                self.covariances[target_id][sensor_id] = np.diag([10.0, 10.0, 2.0, 2.0])

    def _get_neighbors(self, sensor_id: int) -> List[int]:
        neighbors_mat = (self.weight != 0).astype(int)
        np.fill_diagonal(neighbors_mat, 0)
        return [idx for idx, val in enumerate(neighbors_mat[sensor_id]) if val == 1]

    def _measurement_from_distance(self, distance_squared: float, sensor_pos: np.ndarray, 
                                   estimated_pos: np.ndarray) -> np.ndarray:
        direction = estimated_pos[:2] - sensor_pos
        norm = np.linalg.norm(direction)
        
        if norm < 1e-6:
            direction = np.random.randn(2)
            norm = np.linalg.norm(direction)
        
        direction = direction / norm
        
        distance = np.sqrt(max(distance_squared, 0.1))
        measured_pos = sensor_pos + direction * distance
        
        return measured_pos

    def _run_consensus(self, target_id: int) -> None:
        for _ in range(self.consensus_iterations):
            new_states = {}
            new_covariances = {}
            
            for sensor_id in self.sensor_ids:
                neighbors = self._get_neighbors(sensor_id)
                if not neighbors:
                    new_states[sensor_id] = self.state_estimates[target_id][sensor_id].copy()
                    new_covariances[sensor_id] = self.covariances[target_id][sensor_id].copy()
                    continue
                
                neighbor_states = [self.state_estimates[target_id][neighbor] for neighbor in neighbors]
                neighbor_covs = [self.covariances[target_id][neighbor] for neighbor in neighbors]
                
                weights = [abs(self.weight[sensor_id][neighbor]) for neighbor in neighbors]
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                avg_state = np.zeros_like(self.state_estimates[target_id][sensor_id])
                avg_cov = np.zeros_like(self.covariances[target_id][sensor_id])
                
                for w, state, cov in zip(weights, neighbor_states, neighbor_covs):
                    avg_state += w * state
                    avg_cov += w * cov
                
                own_state = self.state_estimates[target_id][sensor_id]
                own_cov = self.covariances[target_id][sensor_id]
                
                new_states[sensor_id] = (1 - self.consensus_weight) * own_state + self.consensus_weight * avg_state
                new_covariances[sensor_id] = (1 - self.consensus_weight) * own_cov + self.consensus_weight * avg_cov
            
            for sensor_id in self.sensor_ids:
                self.state_estimates[target_id][sensor_id] = new_states[sensor_id]
                self.covariances[target_id][sensor_id] = new_covariances[sensor_id]

    def run_main_algorithm(self) -> Dict[int, Dict[int, np.ndarray]]:
        for target_id in self.target_ids:
            for sensor_id in self.sensor_ids:
                x_prev = self.state_estimates[target_id][sensor_id]
                P_prev = self.covariances[target_id][sensor_id]
                
                x_pred = self.F @ x_prev
                P_pred = self.F @ P_prev @ self.F.T + self.Q
                
                self.state_estimates[target_id][sensor_id] = x_pred
                self.covariances[target_id][sensor_id] = P_pred
            
            for sensor_id in self.sensor_ids:
                x_pred = self.state_estimates[target_id][sensor_id]
                P_pred = self.covariances[target_id][sensor_id]
                
                sensor_pos = self.sensors_positions[sensor_id]
                distance_squared = self.distances[target_id][sensor_id]
                
                z_measured_pos = self._measurement_from_distance(
                    distance_squared, sensor_pos, x_pred[:2]
                )
                
                H = self.H
                
                z_pred = self.H @ x_pred
                y = z_measured_pos - z_pred
                
                S = H @ P_pred @ H.T + self.R
                
                try:
                    S_inv = np.linalg.inv(S)
                    K = P_pred @ H.T @ S_inv
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)
                    K = P_pred @ H.T @ S_inv
                
                x_corrected = x_pred + K @ y
                P_corrected = (np.eye(self.state_dim) - K @ H) @ P_pred
                
                self.state_estimates[target_id][sensor_id] = x_corrected
                self.covariances[target_id][sensor_id] = P_corrected
            
            self._run_consensus(target_id)
        
        result: Dict[int, Dict[int, np.ndarray]] = {}
        for target_id in self.target_ids:
            result[target_id] = {}
            for sensor_id in self.sensor_ids:
                result[target_id][sensor_id] = self.state_estimates[target_id][sensor_id][:2].copy()
        
        return result

    def run_n_iterations(self, data: Dict[int, Any]) -> Dict[int, Any]:
        from collections import defaultdict
        
        result = defaultdict()
        for iteration in data.keys():
            self.true_targets_position = data[iteration][0]
            self.distances = data[iteration][1]
            new_estimates = self.run_main_algorithm()
            
            for target_id in self.target_ids:
                for sensor_id in self.sensor_ids:
                    self.state_estimates[target_id][sensor_id][:2] = new_estimates[target_id][sensor_id]
            
            result[iteration] = [data[iteration][0], new_estimates]
        return result