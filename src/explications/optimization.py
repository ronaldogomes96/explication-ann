import numpy as np


def get_otimal_bounds(original_bounds, box_bounds):
    otimal_bounds = []

    for layer_index, layer_box_values in enumerate(box_bounds):
        layer_bounds = []

        for neuron_index, neuron_box_values in enumerate(layer_box_values):
            if layer_index != (len(box_bounds) - 1):
                min = np.maximum(original_bounds['layers'][layer_index][neuron_index][0], neuron_box_values[0])
                max = np.minimum(original_bounds['layers'][layer_index][neuron_index][1], neuron_box_values[1])
                layer_bounds.append((min, max))
            else:
                min = np.maximum(original_bounds['output'][neuron_index][0], neuron_box_values[0])
                max = np.minimum(original_bounds['output'][neuron_index][1], neuron_box_values[1])
                layer_bounds.append((min, max))

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
