import numpy as np

from unittest.mock import patch
from algorithms.original_spsa import Original_SPSA

class TestOriginalSPSA:
    def setup_method(self):
        self.sensors_positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([0.0, 10.0])
        }
        self.true_targets_position = {
            0: np.array([5.0, 5.0]),
            1: np.array([3.0, 7.0])
        }
        self.distances = {
            0: {0: 50.0, 1: 50.0, 2: 50.0},
            1: {0: 58.0, 1: 58.0, 2: 58.0}
        }
        self.init_coords = {
            0: {
                0: np.array([6.0, 4.0]),
                1: np.array([4.0, 6.0]),
                2: np.array([5.0, 5.0])
            },
            1: {
                0: np.array([2.0, 8.0]),
                1: np.array([4.0, 6.0]),
                2: np.array([3.0, 7.0])
            }
        }

    def test_initialization(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position,
            distances=self.distances,
            init_coords=self.init_coords
        )
        assert spsa.number_of_sensors == 3
        assert spsa.number_of_targets == 2
        assert spsa.sensor_ids == {0, 1, 2}
        assert spsa.target_ids == {0, 1}
        assert spsa.dimensions == 2
        assert spsa.beta_1 == 0.5
        assert spsa.beta_2 == 0.5
        assert spsa.beta == 1.0
        assert spsa.alpha == 0.25
        assert spsa.gamma == 0.25
        assert spsa.b == 1
        assert spsa.Delta_abs_value == 1 / np.sqrt(2)

    def test_condition_number(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        test_matrix = np.array([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0]
        ])
        condition_number = spsa._condition_number(test_matrix)
        assert condition_number > 0

    def test_rho_overline(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        result = spsa._rho_overline(10.0, 7.0)
        assert result == 3.0

    def test_compute_error(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        vec1 = np.array([1.0, 2.0])
        vec2 = np.array([4.0, 6.0])
        error = spsa._compute_error(vec1, vec2)
        expected_error = 49.0
        assert error == expected_error

    def test_get_random_neighbors(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        weight_matrix = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        neighbors = spsa._get_random_neibors(weight_matrix, max=2)
        assert 0 in neighbors
        assert 1 in neighbors
        assert 2 in neighbors
        for sensor_id in [0, 1, 2]:
            assert len(neighbors[sensor_id]) <= 2

    def test_calc_D_l_i_j(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        meas_l = {0: 10.0, 1: 8.0, 2: 12.0}
        spsa.s_norms = {0: 0.0, 1: 100.0, 2: 100.0}
        result = spsa._calc_D_l_i_j(meas_l, 0, 1)
        expected = 102.0
        assert result == expected

    @patch('random.random')
    @patch('random.sample')
    def test_run_main_algorithm(self, mock_sample, mock_random):
        mock_random.side_effect = [0.3, 0.7, 0.2, 0.8] * 10
        mock_sample.return_value = [1, 2]
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position,
            distances=self.distances,
            init_coords=self.init_coords
        )
        spsa.weight = np.array([
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0]
        ])
        result = spsa.run_main_algorithm()
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        for target_id in [0, 1]:
            for sensor_id in [0, 1, 2]:
                assert sensor_id in result[target_id]
                assert isinstance(result[target_id][sensor_id], np.ndarray)
                assert len(result[target_id][sensor_id]) == 2

    def test_run_n_iterations(self):
        data = {
            0: [self.true_targets_position, self.distances],
            1: [self.true_targets_position, self.distances],
            2: [self.true_targets_position, self.distances]
        }
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position,
            distances=self.distances,
            init_coords=self.init_coords
        )
        result = spsa.run_n_iterations(data)
        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert 2 in result
        for iteration in [0, 1, 2]:
            assert len(result[iteration]) == 2
            assert 0 in result[iteration][0]
            assert 1 in result[iteration][0]
            assert 0 in result[iteration][1]
            assert 1 in result[iteration][1]

    def test_update_matrix(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        spsa.weight = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        cond, weight = spsa._update_matrix()
        assert cond > 0
        assert weight.shape == (3, 3)
        assert np.array_equal(weight, spsa.weight)

    def test_gen_new_coordinates(self):
        spsa = Original_SPSA(
            sensors_positions=self.sensors_positions,
            true_targets_position=self.true_targets_position
        )
        original_coords = np.array([5.0, 5.0])
        with patch('random.random') as mock_random:
            mock_random.return_value = 0.5
            new_coords = spsa._gen_new_coordinates(original_coords, R=1.0)
            assert isinstance(new_coords, np.ndarray)
            assert len(new_coords) == 2
            assert not np.array_equal(new_coords, original_coords)