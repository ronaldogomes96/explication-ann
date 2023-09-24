import unittest
import numpy as np

from numpy.testing import assert_array_equal

from src.explications.box import box_relax_input_to_bounds, box_forward, box_check_solution, box_has_solution


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

    def test_box_has_solution(self):
        layers = (
            Layer(
                weights=(
                    (0.07841454, -0.8021213, -0.5349561),
                    (-0.872978, -0.7981068, -0.9043029),
                    (1.2508274, -0.6652787, -0.32324892),
                    (2.4814022, 0.03720671, -0.84925544)
                ),
                biases=(-0.40919042, 0.0, 0.0)
            ),
            Layer(
                weights=(
                    (2.7906456, 0.8087422, 0.80422074),
                    (-0.68973327, 0.34983993, -0.826509),
                    (-0.7777126, -0.07366228, -0.05996084)
                ),
                biases=(-0.0019502, -1.0573266, -1.0513506)
            ),
            Layer(
                weights=(
                    (-3.3824484, 1.3298496, 0.5923862),
                    (-0.08571004, -4.973834, 3.4073255),
                    (0.26439068, -4.436257, 3.1741834)
                ),
                biases=(5.100699, -0.4697151, -2.5913954)
            ),
        )
        network_output = 0
        input_bounds = (
            (0.0, 1.0),
            (0.625, 0.625),
            (0.0677966101694915, 0.0677966101694915),
            (0.0416666666666666, 0.0416666666666666)
        )
        self.assertTrue(box_has_solution(input_bounds, layers, network_output))
        input_bounds = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0677966101694915, 0.0677966101694915),
            (0.0416666666666666, 0.0416666666666666)
        )
        self.assertTrue(box_has_solution(input_bounds, layers, network_output))
        input_bounds = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0416666666666666, 0.0416666666666666)
        )
        self.assertFalse(box_has_solution(input_bounds, layers, network_output))
        input_bounds = (
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0677966101694915, 0.0677966101694915),
            (0.0, 1.0)
        )
        self.assertFalse(box_has_solution(input_bounds, layers, network_output))


if __name__ == '__main__':
    unittest.main()
