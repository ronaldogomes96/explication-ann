import numpy as np

from docplex.mp.model import Model

from src.explications.milp import maximize, minimize


def build_tjeng_network(mdl: Model, layers, variables, metrics):
    output_bounds = []
    last_layer = layers[-1]
    for layer_index, layer in enumerate(layers):
        x = variables['input'] if layer_index == 0 else variables['intermediate'][layer_index - 1]
        _A = layer.get_weights()[0].T
        _b = layer.get_weights()[1]
        _y, _z = (variables['intermediate'][layer_index], variables['decision'][layer_index]) if layer != last_layer \
            else (variables['output'], np.empty(len(_A)))
        for neuron_index, (A, b, y, z) in enumerate(zip(_A, _b, _y, _z)):
            result = A @ x + b
            upper_bound = maximize(mdl, result)
            if upper_bound <= 0 and layer != last_layer:
                mdl.add_constraint(y == 0, ctname=f'c_{layer_index}_{neuron_index}')
                metrics['constraints'] += 1
                continue
            lower_bound = minimize(mdl, result)
            if lower_bound >= 0 and layer != last_layer:
                mdl.add_constraint(y == result, ctname=f'c_{layer_index}_{neuron_index}')
                metrics['constraints'] += 1
                continue
            if layer != last_layer:
                mdl.add_constraint(y <= result - lower_bound * (1 - z))
                mdl.add_constraint(y >= result)
                mdl.add_constraint(y <= upper_bound * z)
                metrics['constraints'] += 3
            else:
                mdl.add_constraint(y == result)
                metrics['constraints'] += 1
                output_bounds.append((lower_bound, upper_bound))
    return np.array(output_bounds)


def insert_tjeng_output_constraints(mdl: Model, output_bounds, variables, network_output):
    output_variable = variables['output'][network_output]
    upper_lower_diffs = output_bounds[network_output][1] - np.array(output_bounds)[:, 0]
    binary_index = 0
    for output_index, output in enumerate(variables['output']):
        if output_index == network_output:
            continue
        diff = upper_lower_diffs[output_index]
        binary_variable = variables['binary'][binary_index]
        mdl.add_constraint(output_variable - output - diff * (1 - binary_variable) <= 0)
        binary_index += 1
