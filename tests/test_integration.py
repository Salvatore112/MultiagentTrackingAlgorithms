import numpy as np

from unittest.mock import patch
from simulations.simulation import Simulation, TARGET_TYPE
from algorithms.original_spsa import Original_SPSA

class TestIntegration:
    def test_simulation_to_spsa_integration(self):
        sim = Simulation(duration=5, time_step=1.0)
        sim.add_sensor(0, (0.0, 0.0))
        sim.add_sensor(1, (10.0, 0.0))
        sim.add_target(0, (5.0, 5.0), (1.0, 1.0), TARGET_TYPE.LINEAR)
        sim.run_simulation()
        spsa_data = sim.get_spsa_input_data()
        spsa = Original_SPSA(
            sensors_positions=spsa_data['sensors_positions'],
            true_targets_position=spsa_data['data'][0][0],
            distances=spsa_data['data'][0][1],
            init_coords=spsa_data['init_coords']
        )
        result = spsa.run_n_iterations(spsa_data['data'])
        assert isinstance(result, dict)
        assert len(result) > 0
        for iteration_data in result.values():
            assert len(iteration_data) == 2
            assert isinstance(iteration_data[0], dict)
            assert isinstance(iteration_data[1], dict)
