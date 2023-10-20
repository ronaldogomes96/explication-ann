import numpy as np


def box_relax_input_to_bounds(network_input, input_bounds, relax_input_mask):
    relaxed_input_bounds = np.reshape(network_input, (-1, 1)).repeat(2, axis=1)
    relaxed_input_bounds[relax_input_mask] = input_bounds[relax_input_mask]
    return relaxed_input_bounds


def box_forward(input_bounds, input_weights, input_biases, apply_relu=True):
    input_bounds = np.concatenate((input_bounds, np.flip(input_bounds, axis=1)), axis=0)
    input_weights = np.concatenate((np.maximum(input_weights, 0), np.minimum(input_weights, 0)), axis=1)
    input_biases = np.reshape(input_biases, (-1, 1))
    output_bounds = np.dot(input_weights, input_bounds) + input_biases
    return np.maximum(output_bounds, 0) if apply_relu else output_bounds


def box_check_solution(output_bounds, network_output):
    lower_bound = output_bounds[network_output][0]
    output_bounds = np.delete(output_bounds, network_output, axis=0)
    max_upper_bound = np.max(output_bounds, axis=0)[1]
    return lower_bound > max_upper_bound


def box_has_solution(bounds, layers, network_output):
    last_layer = layers[-1]
    for layer in layers:
        weights = layer.get_weights()[0].T
        biases = layer.get_weights()[1]
        bounds = box_forward(bounds, weights, biases) if layer != last_layer \
            else box_forward(bounds, weights, biases, apply_relu=False)
    return box_check_solution(bounds, network_output)
