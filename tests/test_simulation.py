import pytest
import numpy as np
import tempfile

from unittest.mock import patch
from simulations.simulation import Simulation, Target, TARGET_TYPE


class TestSimulation:
    def setup_method(self):
        self.sim = Simulation(duration=10, time_step=1.0)

    def test_simulation_initialization(self):
        assert self.sim.duration == 10
        assert self.sim.time_step == 1.0
        assert len(self.sim.sensors) == 0
        assert len(self.sim.targets) == 0
        assert self.sim.simulation_data == {}
        assert "simulation_results" in self.sim.output_dir

    def test_add_sensor(self):
        self.sim.add_sensor(1, (10.0, 20.0))
        assert len(self.sim.sensors) == 1
        sensor = self.sim.sensors[0]
        assert sensor.id == 1
        assert sensor.position == (10.0, 20.0)

    def test_add_target(self):
        self.sim.add_target(1, (5.0, 5.0), (1.0, 1.0), TARGET_TYPE.LINEAR)
        assert len(self.sim.targets) == 1
        target = self.sim.targets[0]
        assert target.id == 1
        assert target.initial_position == (5.0, 5.0)
        assert target.velocity == (1.0, 1.0)
        assert target.movement_type == TARGET_TYPE.LINEAR

    def test_add_linear_target(self):
        with patch("random.uniform") as mock_uniform, patch(
            "random.getstate"
        ) as mock_getstate, patch("random.setstate") as mock_setstate:
            mock_uniform.side_effect = [10.0, 20.0, 2.0, 0.0]
            mock_getstate.return_value = "random_state"
            initial_pos, velocity = self.sim.add_linear_target(1, area_size=50)
            assert len(self.sim.targets) == 1
            target = self.sim.targets[0]
            assert target.id == 1
            assert target.movement_type == TARGET_TYPE.LINEAR
            assert target.random_walk_params is None
            assert mock_setstate.called

    def test_add_random_walk_target(self):
        with patch("random.uniform") as mock_uniform, patch(
            "random.getstate"
        ) as _, patch("random.setstate") as _:
            mock_uniform.side_effect = [10.0, 20.0, 1.5, 0.0, 0.2, 0.15, 0.7]
            initial_pos, velocity = self.sim.add_random_walk_target(1, area_size=50)
            assert len(self.sim.targets) == 1
            target = self.sim.targets[0]
            assert target.id == 1
            assert target.movement_type == TARGET_TYPE.RANDOM_WALK
            assert target.random_walk_params is not None
            assert "speed_variation" in target.random_walk_params

    def test_add_uniform_sensor(self):
        with patch("random.uniform") as mock_uniform, patch(
            "random.getstate"
        ) as mock_getstate, patch("random.setstate") as mock_setstate:
            mock_uniform.return_value = 15.0
            mock_getstate.return_value = "random_state"
            position = self.sim.add_uniform_sensor(1, area_size=50)
            assert len(self.sim.sensors) == 1
            sensor = self.sim.sensors[0]
            assert sensor.id == 1
            assert position == (15.0, 15.0)
            assert mock_setstate.called

    def test_calculate_distance(self):
        pos1 = (0.0, 0.0)
        pos2 = (3.0, 4.0)
        distance = self.sim.calculate_distance(pos1, pos2)
        assert distance == 25.0

    def test_get_linear_position(self):
        target = Target(1, (0.0, 0.0), (2.0, 1.0), TARGET_TYPE.LINEAR)
        position = self.sim._get_linear_position(target, 2.0)
        assert position == (4.0, 2.0)

    def test_get_target_position_linear(self):
        target = Target(1, (0.0, 0.0), (1.0, 2.0), TARGET_TYPE.LINEAR)
        position = self.sim.get_target_position(target, 3.0)
        assert position == (3.0, 6.0)

    def test_run_simulation(self):
        self.sim.add_sensor(1, (0.0, 0.0))
        self.sim.add_target(1, (1.0, 1.0), (0.5, 0.5), TARGET_TYPE.LINEAR)
        self.sim.run_simulation()
        assert "time_points" in self.sim.simulation_data
        assert "sensors" in self.sim.simulation_data
        assert "targets" in self.sim.simulation_data
        time_points = self.sim.simulation_data["time_points"]
        assert len(time_points) == 11
        assert time_points[0] == 0.0
        assert time_points[-1] == 10.0

    def test_get_distance(self):
        self.sim.add_sensor(1, (0.0, 0.0))
        self.sim.add_target(1, (3.0, 4.0), (0.0, 0.0), TARGET_TYPE.LINEAR)
        self.sim.run_simulation()
        distance = self.sim.get_distance(1, 1, 0.0)
        assert distance == 25.0

    def test_get_target_position_at_time(self):
        self.sim.add_target(1, (1.0, 2.0), (1.0, 1.0), TARGET_TYPE.LINEAR)
        self.sim.run_simulation()
        position = self.sim.get_target_position_at_time(1, 2.0)
        assert position == (3.0, 4.0)

    def test_save_and_load_simulation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.sim.output_dir = temp_dir
            self.sim.add_sensor(1, (0.0, 0.0))
            self.sim.add_target(1, (1.0, 1.0), (0.5, 0.5), TARGET_TYPE.LINEAR)
            self.sim.run_simulation()
            self.sim.save_simulation("test_simulation.pkl")
            new_sim = Simulation(duration=5, time_step=0.5)
            new_sim.output_dir = temp_dir
            new_sim.load_simulation("test_simulation.pkl")
            assert new_sim.duration == self.sim.duration
            assert new_sim.time_step == self.sim.time_step
            assert len(new_sim.sensors) == len(self.sim.sensors)
            assert len(new_sim.targets) == len(self.sim.targets)

    def test_get_spsa_input_data(self):
        self.sim.add_sensor(1, (0.0, 0.0))
        self.sim.add_target(1, (1.0, 1.0), (0.5, 0.5), TARGET_TYPE.LINEAR)
        self.sim.run_simulation()
        spsa_data = self.sim.get_spsa_input_data()
        assert "sensors_positions" in spsa_data
        assert "init_coords" in spsa_data
        assert "data" in spsa_data
        assert 1 in spsa_data["sensors_positions"]
        assert np.array_equal(spsa_data["sensors_positions"][1], np.array([0.0, 0.0]))
        assert 1 in spsa_data["init_coords"]
        assert 1 in spsa_data["init_coords"][1]
        assert 0 in spsa_data["data"]
        assert len(spsa_data["data"][0]) == 2

    def test_print_simulation_info(self):
        self.sim.add_sensor(1, (0.0, 0.0))
        self.sim.add_target(1, (1.0, 1.0), (0.5, 0.5), TARGET_TYPE.LINEAR)
        self.sim.print_simulation_info()

    def test_invalid_time_raises_error(self):
        self.sim.add_sensor(1, (0.0, 0.0))
        self.sim.add_target(1, (1.0, 1.0), (0.5, 0.5), TARGET_TYPE.LINEAR)
        self.sim.run_simulation()
        with pytest.raises(ValueError):
            self.sim.get_distance(1, 1, -1.0)
        with pytest.raises(ValueError):
            self.sim.get_distance(1, 1, 15.0)

    def test_target_type_enum(self):
        assert str(TARGET_TYPE.LINEAR) == "lin"
        assert str(TARGET_TYPE.RANDOM_WALK) == "ran_wlk"
        linear_target = TARGET_TYPE.LINEAR
        random_target = TARGET_TYPE.RANDOM_WALK
        assert linear_target != random_target
        assert linear_target.value == 1
        assert random_target.value == 2
