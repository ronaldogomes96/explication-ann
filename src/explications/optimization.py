import numpy as np


def get_otimal_bounds(original_bounds, box_bounds, metrics):
    otimal_bounds = []

    for layer_index, layer_box_values in enumerate(box_bounds):
        layer_bounds = []

        for neuron_index, neuron_box_values in enumerate(layer_box_values):
            lower = original_bounds['layers'][layer_index] if layer_index != (len(box_bounds) - 1) else original_bounds['output']

            min = np.maximum(lower[neuron_index][0], neuron_box_values[0])
            max = np.minimum(lower[neuron_index][1], neuron_box_values[1])

            layer_bounds.append((min, max))

            metrics['times_box_optimization_calculated'] += 1

            if (min > lower[neuron_index][0]) or (max < lower[neuron_index][1]):
                metrics['times_box_optimize_some_bounds'] += 1

                if (min > lower[neuron_index][0]) and (max < lower[neuron_index][1]):
                    metrics['times_box_optimize_two_bounds'] += 1

        otimal_bounds.append(layer_bounds)

    return otimal_bounds


def get_tjeng_variables(mdl_box, initial_bounds, layers):
    input_variables = []

    for index_input, _ in enumerate(zip(initial_bounds['input'])):
        name = f'x_{index_input}'
        input_variables.append(mdl_box.get_var_by_name(name))

    tjeng_variables = {
        'input': input_variables,
        'intermediate': [],
        'decision': []
    }

    last_layer = layers[-1]

    for layer_index, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]

        tjeng_output = []
        tjeng_intermediate = []
        tjeng_decision = []

        for actual in range(number_variables):
            if layer == last_layer:
                tjeng_output.append(mdl_box.get_var_by_name(f'o_{actual}'))

            tjeng_intermediate.append(mdl_box.get_var_by_name(f'y_{layer_index}_{actual}'))
            tjeng_decision.append(mdl_box.get_var_by_name(f'z_{layer_index}_{actual}'))

        if layer == last_layer:
            tjeng_variables['output'] = tjeng_output
        else:
            tjeng_variables['intermediate'].append(tjeng_intermediate)
            tjeng_variables['decision'].append(tjeng_decision)

    return tjeng_variables
