import numpy as np

from random import random, sample
from collections import defaultdict


class Original_SPSA:
    def __init__(
        self,
        sensors_positions=None,
        true_targets_position=None,
        distances=None,
        init_coords=None,
    ):
        self.number_of_sensors = len(sensors_positions.items())
        self.sensor_ids = {i for i in range(self.number_of_sensors)}
        self.sensors_positions = sensors_positions
        self.number_of_targets = len(true_targets_position.keys())
        self.target_ids = {i for i in range(self.number_of_targets)}
        self.true_targets_position = true_targets_position
        self.distances = distances
        # Coordinates from the previous iteration of some random coordinates for the 1st iteration
        self.init_coords = init_coords

        # Algorithm specific variables from the paper
        self.dimensions = 2
        self.beta_1 = 0.5
        self.beta_2 = 0.5
        self.beta = self.beta_1 + self.beta_2
        self.alpha = 0.25
        self.gamma = 0.25
        self.b = 1
        self.s_norms = {i: sum(val * val) for i, val in self.sensors_positions.items()}
        self.theta = np.array([val for _, val in self.true_targets_position.items()])
        self.Delta_abs_value = 1 / np.sqrt(self.dimensions)

        max_condition_number = 0
        while max_condition_number < 3:
            weight = self._generate_matrix(self.number_of_sensors)
            condition_number = self._condition_number(weight)
            if condition_number > max_condition_number:
                max_condition_number = condition_number
                self.weight = weight

    def _generate_matrix(self, n):
        raw_mat = np.random.rand(n, n)
        weight = np.tril(raw_mat) + np.tril(raw_mat, -1).T
        weight = np.around(weight, 1)
        np.fill_diagonal(weight, 0)
        weight = -weight + np.diag([n - 1] * n)
        return weight

    def _condition_number(self, matrix):
        eig = np.linalg.eig(matrix)[0]
        eig = sorted([abs(n) for n in eig if abs(n) > 0.00001])
        return eig[-1] / eig[0]

    def _rho_overline(self, meas_1: float, meas_2: float):
        return meas_1 - meas_2

    def _update_matrix(self):
        weight = self.weight
        cond = self._condition_number(weight)
        return cond, weight

    def run_main_algorithm(self):
        _, weight = self._update_matrix()

        errors = {}

        theta_hat = {
            target_id: {
                sensor_id: self.init_coords[target_id][sensor_id].copy()
                for sensor_id in self.sensor_ids
            }
            for target_id in self.target_ids
        }

        theta_new = {}
        err = 0

        for l in self.target_ids:  # noqa: E741
            theta_new[l] = {}
            neighbors = self._get_random_neibors(weight, 2)

            for i in self.sensor_ids:
                coef1 = 1 if random() < 0.5 else -1
                coef2 = 1 if random() < 0.5 else -1
                delta = np.array(
                    [coef1 * self.Delta_abs_value, coef2 * self.Delta_abs_value]
                )

                x1 = theta_hat[l][i] + self.beta_1 * delta
                x2 = theta_hat[l][i] - self.beta_2 * delta

                y1 = self._f_l_i(l, i, x1, neighbors)
                y2 = self._f_l_i(l, i, x2, neighbors)

                spsa = (y1 - y2) / self.beta * delta / 2

                neighbors_i = neighbors.get(i, [])
                b = weight[i]
                theta_diff = [
                    abs(b[j]) * (theta_hat[l][i] - theta_hat[l][j]) for j in neighbors_i
                ]

                theta_new[l][i] = theta_hat[l][i] - (
                    self.alpha * spsa + self.gamma * sum(theta_diff)
                )

                err += self._compute_error(
                    theta_new[l][i], self.true_targets_position[l]
                )

        theta_hat = theta_new.copy()

        self.errors = errors

        target_err = {}
        for target_id in self.target_ids:
            target_err[target_id] = {
                sensor_id: self._compute_error(
                    theta_hat[target_id][sensor_id],
                    self.true_targets_position[target_id],
                )
                for sensor_id in self.sensor_ids
            }

        return theta_hat

    def _f_l_i(self, l, i, r_hat_l, neibors):
        C = self._C_i(i, neibors)
        D = self._D_l_i(l, i, neibors)

        try:
            C_i_inv = np.linalg.inv(C)
        except Exception:
            C_i_inv = np.linalg.pinv(C)

        diff = r_hat_l - np.matmul(C_i_inv, D)
        return sum(diff * diff)

    def _C_i(self, i, neibors):
        C_i = [
            self.sensors_positions.get(j) - self.sensors_positions.get(i)
            for j in neibors.get(i)
        ]
        return 2 * np.array(C_i)

    def _D_l_i(self, l, i, neibors):
        Dli = [self._calc_D_l_i_j(self.distances.get(l), i, j) for j in neibors.get(i)]
        return Dli

    def _calc_D_l_i_j(self, meas_l: dict, i, j):
        return (
            self._rho_overline(meas_l.get(i), meas_l.get(j))
            + self.s_norms.get(j)
            - self.s_norms.get(i)
        )

    def _gen_new_coordinates(self, coords: np.array, R: float = 1):
        phi = 2 * np.pi * random()
        rad = R * random()

        shift = rad * np.array([np.sin(phi), np.cos(phi)])
        return coords + shift

    def _compute_error(self, vector_1, vector_2):
        return pow(sum(vector_1 - vector_2), 2)

    def _get_random_neibors(self, weight, max=2):
        neibors_mat = (weight != 0).astype(int)
        np.fill_diagonal(neibors_mat, 0)

        neibors = {}
        for sensor_id in self.sensor_ids:
            neib = [ind for ind, sens in enumerate(neibors_mat[sensor_id]) if sens == 1]
            if len(neib) > max:
                neib = sample(neib, max)
            neibors[sensor_id] = neib

        return neibors

    def run_n_iterations(self, data):
        result = defaultdict()
        for iteration in data.keys():
            self.true_targets_position = data[iteration][0]
            self.distances = data[iteration][1]
            new_estimates = self.run_main_algorithm()
            self.init_coords = new_estimates
            result[iteration] = [data[iteration][0], new_estimates]
        return result
