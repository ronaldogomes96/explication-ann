import unittest
import numpy as np

from numpy.testing import assert_array_equal

from src.explications.box import box_relax_input_to_bounds, box_forward, box_check_solution


class Layer:
    def __init__(self, weights, biases):
        self._weights = [np.array(weights), np.array(biases)]

    def get_weights(self):
        return self._weights


class TestBox(unittest.TestCase):
    def test_box_relax_input_to_bounds(self):
        network_input = (0.2222222222222221, 0.625, 0.0677966101694915, 0.0416666666666666)
        input_bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        relax_input_mask = (True, False, True, False)
        relaxed_input_bounds = box_relax_input_to_bounds(network_input, input_bounds, relax_input_mask)
        expected_relaxed_input_bounds = np.array((
            (0.0, 1.0),
            (0.625, 0.625),
            (0.0, 1.0),
            (0.0416666666666666, 0.0416666666666666)
        ))
        assert_array_equal(relaxed_input_bounds, expected_relaxed_input_bounds)

    def test_box_forward_without_relu(self):
        input_bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        input_weights = (
            (0.07841454, -0.872978, 1.2508274, 2.4814022),
            (-0.8021213, -0.7981068, -0.6652787, 0.03720671),
            (-0.5349561, -0.9043029, -0.32324892, -0.84925544)
        )
        input_biases = (-0.40919042, 0.0, 0.0)
        output_bounds = box_forward(input_bounds, input_weights, input_biases, apply_relu=False)
        expected_output_bounds = np.array((
            (-1.28216842, 3.40145372),
            (-2.2655068, 0.03720671),
            (-2.61176336, 0.0)
        ))
        assert_array_equal(output_bounds, expected_output_bounds)

    def test_box_forward_with_relu(self):
        input_bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        input_weights = (
            (0.07841454, -0.872978, 1.2508274, 2.4814022),
            (-0.8021213, -0.7981068, -0.6652787, 0.03720671),
            (-0.5349561, -0.9043029, -0.32324892, -0.84925544)
        )
        input_biases = (-0.40919042, 0.0, 0.0)
        output_bounds = box_forward(input_bounds, input_weights, input_biases)
        expected_output_bounds = np.array((
            (0.0, 3.40145372),
            (0.0, 0.03720671),
            (0.0, 0.0)
        ))
        assert_array_equal(output_bounds, expected_output_bounds)

    def test_box_check_solution(self):
        network_output = 0
        output_bounds = (
            (5.100699, 5.100699),
            (-0.4697151, -0.4697151),
            (-2.5913954, -2.5913954)
        )
        self.assertTrue(box_check_solution(output_bounds, network_output))
        output_bounds = (
            (-4.55320543, 5.100699),
            (-0.4697151, 3.32583163),
            (-2.5913954, -0.90065571)
        )
        self.assertFalse(box_check_solution(output_bounds, network_output))


if __name__ == '__main__':
    unittest.main()
