import unittest
import numpy as np

from numpy.testing import assert_array_equal

from src.explications.box import box_relax_input_bounds


class Layer:
    def __init__(self, weights, biases):
        self._weights = [np.array(weights), np.array(biases)]

    def get_weights(self):
        return self._weights


class TestBox(unittest.TestCase):
    def test_box_relax_input_bounds(self):
        input_bounds = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        network_input = (0.2222222222222221, 0.625, 0.0677966101694915, 0.0416666666666666)
        relax_input_mask = (True, False, True, False)
        relaxed_input_bounds = box_relax_input_bounds(input_bounds, network_input, relax_input_mask)
        expected_relaxed_input_bounds = np.array((
            (0.0, 1.0),
            (0.625, 0.625),
            (0.0, 1.0),
            (0.0416666666666666, 0.0416666666666666)
        ))
        assert_array_equal(relaxed_input_bounds, expected_relaxed_input_bounds)


if __name__ == '__main__':
    unittest.main()
