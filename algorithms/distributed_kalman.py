import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple
from random import random


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

        self.T: float = 0.1
        self.m: int = 4
        
        self.F = np.array([
            [1, 0, self.T, 0],
            [0, 1, 0, self.T],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.G = np.zeros((4, 4))
        self.Q = 0.1 * np.eye(4)
        self.d = np.array([0, 0, 0, -9.8 * self.T]).reshape(4, 1)
        
        self.sigma = 0.75 * np.random.rand(self.number_of_sensors)
        self.R_matrices: Dict[int, np.ndarray] = {}
        for i in range(self.number_of_sensors):
            self.R_matrices[i] = self.sigma[i] * np.eye(2)
        
        self.algebraic_connectivity = 0
        while self.algebraic_connectivity < 0.01:
            self.A, self.spectral_rad, self.algebraic_connectivity, self.degree_vector, self.coordinates = self._graph_properties(self.number_of_sensors)
        
        self.W = self._compute_metropolis_weights(self.A, self.degree_vector)
        
        self.x_minus: Dict[int, Dict[int, np.ndarray]] = {}
        self.x_plus: Dict[int, Dict[int, np.ndarray]] = {}
        self.P_minus: Dict[int, Dict[int, np.ndarray]] = {}
        self.P_plus: Dict[int, Dict[int, np.ndarray]] = {}
        
        self.initialize_filters()

    def _graph_properties(self, num_nodes: int) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
        N = num_nodes
        A = np.zeros((N, N))
        radius = 0.5
        
        x_coordinates = np.random.rand(1, N) + 0.1
        y_coordinates = np.random.rand(1, N) + 0.1
        coordinates = np.column_stack((x_coordinates.T, y_coordinates.T))
        
        for k in range(N):
            for l in range(N):
                d = np.sqrt((x_coordinates[0, k] - x_coordinates[0, l])**2 + 
                           (y_coordinates[0, k] - y_coordinates[0, l])**2)
                if d <= radius:
                    A[k, l] = 1
        
        np.fill_diagonal(A, 0)
        
        num_nb = np.sum(A, axis=1)
        degree_vector = num_nb
        
        L = np.diag(num_nb) - A
        sigma = np.linalg.svd(L, compute_uv=False)
        algebraic_connectivity = sigma[-2] if len(sigma) >= 2 else 0
        
        spectral_rad = np.max(np.abs(np.linalg.eig(A)[0]))
        
        return A, spectral_rad, algebraic_connectivity, degree_vector, coordinates

    def _compute_metropolis_weights(self, A: np.ndarray, degree_vector: np.ndarray) -> np.ndarray:
        N = A.shape[0]
        W = np.zeros((N, N))
        
        delta = 0.01
        
        for k in range(N):
            for l in range(N):
                if k == l:
                    W[k, l] = 1 + delta - degree_vector[k] * delta
                elif A[k, l] > 0:
                    W[k, l] = delta
        
        return W

    def initialize_filters(self) -> None:
        for target_id in self.target_ids:
            self.x_minus[target_id] = {}
            self.x_plus[target_id] = {}
            self.P_minus[target_id] = {}
            self.P_plus[target_id] = {}
            
            for sensor_id in self.sensor_ids:
                init_pos = self.init_coords[target_id][sensor_id]
                init_state = np.array([init_pos[0], init_pos[1], 0.0, 0.0]).reshape(4, 1)
                self.x_minus[target_id][sensor_id] = init_state
                self.x_plus[target_id][sensor_id] = init_state
                self.P_minus[target_id][sensor_id] = np.eye(4)
                self.P_plus[target_id][sensor_id] = np.eye(4)

    def _measurement_from_distance(self, distance_squared: float, sensor_pos: np.ndarray, estimated_pos: np.ndarray) -> np.ndarray:
        direction = estimated_pos[:2] - sensor_pos
        norm = np.linalg.norm(direction)
        
        if norm < 1e-6:
            direction = np.random.randn(2)
            norm = np.linalg.norm(direction)
        
        direction = direction / norm
        
        distance = np.sqrt(max(distance_squared, 0.1))
        measured_pos = sensor_pos + direction * distance
        
        return measured_pos

    def _generate_measurements(self, true_positions: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        measurements = {}
        
        for target_id, true_pos in true_positions.items():
            measurements[target_id] = {}
            for sensor_id in self.sensor_ids:
                measured_pos = true_pos + np.sqrt(self.sigma[sensor_id]) * np.random.randn(2)
                measurements[target_id][sensor_id] = measured_pos
        
        return measurements

    def run_main_algorithm(self) -> Dict[int, Dict[int, np.ndarray]]:
        true_positions = self.true_targets_position
        
        measurements = self._generate_measurements(true_positions)
        
        x_local: Dict[int, Dict[int, np.ndarray]] = {}
        P_local: Dict[int, Dict[int, np.ndarray]] = {}
        
        for target_id in self.target_ids:
            x_local[target_id] = {}
            P_local[target_id] = {}
            
            for sensor_id in self.sensor_ids:
                x_local[target_id][sensor_id] = self.x_minus[target_id][sensor_id].copy()
                P_local[target_id][sensor_id] = self.P_minus[target_id][sensor_id].copy()
                
                neighbors = np.where(self.W[:, sensor_id] > 0)[0]
                
                for neighbor_id in neighbors:
                    if neighbor_id == sensor_id:
                        continue
                    
                    z = measurements[target_id][neighbor_id].reshape(2, 1)
                    H = self.H
                    R = self.R_matrices[neighbor_id]
                    P_loc = P_local[target_id][sensor_id]
                    
                    error = z - H @ x_local[target_id][sensor_id]
                    S = R + H @ P_loc @ H.T
                    
                    try:
                        S_inv = np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S)
                    
                    K = P_loc @ H.T @ S_inv
                    x_local[target_id][sensor_id] = x_local[target_id][sensor_id] + K @ error
                    P_local[target_id][sensor_id] = P_loc - K @ H @ P_loc
        
        for target_id in self.target_ids:
            for sensor_id in self.sensor_ids:
                self.P_plus[target_id][sensor_id] = P_local[target_id][sensor_id].copy()
                self.P_minus[target_id][sensor_id] = self.F @ self.P_plus[target_id][sensor_id] @ self.F.T + self.Q
        
        for target_id in self.target_ids:
            for sensor_id in self.sensor_ids:
                x_plus_sum = np.zeros((4, 1))
                
                for l in range(self.number_of_sensors):
                    weight = self.W[l, sensor_id]
                    
                    if weight > 0:
                        x_plus_sum += weight * x_local[target_id][l]
                
                self.x_plus[target_id][sensor_id] = x_plus_sum
                self.x_minus[target_id][sensor_id] = self.F @ self.x_plus[target_id][sensor_id] + self.d
        
        result: Dict[int, Dict[int, np.ndarray]] = {}
        for target_id in self.target_ids:
            result[target_id] = {}
            for sensor_id in self.sensor_ids:
                result[target_id][sensor_id] = self.x_plus[target_id][sensor_id][:2].flatten().copy()
        
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
                    new_state = np.zeros((4, 1))
                    new_state[:2, 0] = new_estimates[target_id][sensor_id]
                    new_state[2:, 0] = self.x_plus[target_id][sensor_id][2:, 0]
                    self.x_plus[target_id][sensor_id] = new_state.copy()
                    self.x_minus[target_id][sensor_id] = self.F @ new_state + self.d
            
            result[iteration] = [data[iteration][0], new_estimates]
        
        return result