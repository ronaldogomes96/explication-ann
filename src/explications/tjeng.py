import numpy as np

from docplex.mp.model import Model

from src.explications.milp import maximize, minimize


def build_tjeng_network(mdl: Model, layers, variables, metrics, otimized_bounds=None):
    output_bounds = []
    layers_bounds = []
    last_layer = layers[-1]

    for layer_index, layer in enumerate(layers):
        layer_bounds = []

        x = variables['input'] if layer_index == 0 else variables['intermediate'][layer_index - 1]
        _A = layer.get_weights()[0].T
        _b = layer.get_weights()[1]
        _y, _z = (variables['intermediate'][layer_index], variables['decision'][layer_index]) if layer != last_layer \
            else (variables['output'], np.empty(len(_A)))

        for neuron_index, (A, b, y, z) in enumerate(zip(_A, _b, _y, _z)):
            result = A @ x + b

            if otimized_bounds is None:
                upper_bound = maximize(mdl, result)
                lower_bound = minimize(mdl, result)

                layer_bounds.append((lower_bound, upper_bound)) if layer != last_layer \
                    else output_bounds.append((lower_bound, upper_bound))

                metrics['accumulated_calls_build_network_without_bounds'] += 1
            else:
                lower_bound = otimized_bounds[layer_index][neuron_index][0]
                upper_bound = otimized_bounds[layer_index][neuron_index][1]

                metrics['accumulated_calls_build_network_with_optimizated_bounds'] += 1

            var_name = 'accumulated_calls_ideal_constraints_without_bounds' if otimized_bounds is None else 'accumulated_calls_ideal_constraints_with_optimizated_bounds'

            if upper_bound <= 0 and layer != last_layer:
                if otimized_bounds is None:
                    mdl.add_constraint(y == 0, ctname=f'c_{layer_index}_{neuron_index}')
                else:
                    mdl.add_user_cut_constraint(y == 0, name=f'c_{layer_index}_{neuron_index}')
                metrics['constraints'] += 1
                metrics[var_name] += 1
                continue

            if lower_bound >= 0 and layer != last_layer:
                if otimized_bounds is None:
                    mdl.add_constraint(y == result, ctname=f'c_{layer_index}_{neuron_index}')
                else:
                    mdl.add_user_cut_constraint(y == result, name=f'c_{layer_index}_{neuron_index}')
                metrics['constraints'] += 1
                metrics[var_name] += 1
                continue

            if layer != last_layer:
                if otimized_bounds is None:
                    mdl.add_constraint(y <= result - lower_bound * (1 - z))
                    mdl.add_constraint(y >= result)
                    mdl.add_constraint(y <= upper_bound * z)
                else:
                    mdl.add_user_cut_constraint(y <= result - lower_bound * (1 - z))
                    mdl.add_user_cut_constraint(y >= result)
                    mdl.add_user_cut_constraint(y <= upper_bound * z)
                metrics['constraints'] += 3
            else:
                if otimized_bounds is None:
                    mdl.add_constraint(y == result)
                else:
                    mdl.add_user_cut_constraint(y == result)
                metrics['constraints'] += 1

            var_name = 'accumulated_calls_binary_constraints_without_bounds' if otimized_bounds is None else 'accumulated_calls_binary_constraints_with_optimizated_bounds'
            metrics[var_name] += 1

        if layer != last_layer:
            layers_bounds.append(layer_bounds)

    return layers_bounds, np.array(output_bounds)


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
