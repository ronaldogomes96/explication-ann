import numpy as np
import pandas as pd

from docplex.mp.model import Model
from time import time

from src.datasets.utils import read_all_datasets
from src.models.utils import load_model


def get_input_variables_and_bounds(mdl: Model, x):
    input_variables = []
    input_bounds = []
    for column_index, column in enumerate(x.columns):
        unique_values = x[column].unique()
        lower_bound, upper_bound = unique_values.min(), unique_values.max()
        name = f'x_{column_index}'
        if len(unique_values) == 2:
            input_variables.append(mdl.binary_var(name=name))
        elif np.any(unique_values.astype('int64') != unique_values.astype('float64')):
            input_variables.append(mdl.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
        else:
            input_variables.append(mdl.integer_var(lb=lower_bound, ub=upper_bound, name=name))
        input_bounds.append((lower_bound, upper_bound))
    return input_variables, input_bounds


def get_intermediate_variables(mdl: Model, layer_index, number_neurons):
    return mdl.continuous_var_list(number_neurons, name='y', key_format=f'_{layer_index}_%s')


def get_decision_variables(mdl: Model, layer_index, number_neurons):
    return mdl.binary_var_list(number_neurons, name='z', key_format=f'_{layer_index}_%s')


def get_output_variables(mdl: Model, number_outputs):
    return mdl.continuous_var_list(number_outputs, lb=-mdl.infinity, name='o')


def maximize(mdl: Model, variable):
    mdl.maximize(variable)
    mdl.solve()
    objective = mdl.objective_value
    mdl.remove_objective()
    return objective


def minimize(mdl: Model, variable):
    mdl.minimize(variable)
    mdl.solve()
    objective = mdl.objective_value
    mdl.remove_objective()
    return objective


def build_tjeng_network(mdl: Model, layers, variables):
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
                continue
            lower_bound = minimize(mdl, result)
            if lower_bound >= 0 and layer != last_layer:
                mdl.add_constraint(y == result, ctname=f'c_{layer_index}_{neuron_index}')
                continue
            if layer != last_layer:

                mdl.add_constraint(y <= result - lower_bound * (1 - z))
                mdl.add_constraint(y >= result)
                mdl.add_constraint(y <= upper_bound * z)
            else:
                mdl.add_constraint(y == result)
                output_bounds.append((lower_bound, upper_bound))
    return output_bounds


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


def build_network(x, layers):
    mdl = Model(name='original')
    variables = {'decision': [], 'intermediate': []}
    bounds = {}
    variables['input'], bounds['input'] = get_input_variables_and_bounds(mdl, x)
    last_layer = layers[-1]
    for layer_index, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]
        if layer == last_layer:
            variables['output'] = get_output_variables(mdl, number_variables)
            break
        variables['intermediate'].append(get_intermediate_variables(mdl, layer_index, number_variables))
        variables['decision'].append(get_decision_variables(mdl, layer_index, number_variables))
    bounds['output'] = build_tjeng_network(mdl, layers, variables)
    return mdl, bounds


def minimal_explication(mdl: Model, bounds, network):
    mdl_clone = mdl.clone(new_name='clone')
    number_features = len(bounds['input'])
    number_outputs = len(bounds['output'])
    variables = {
        'input': [mdl_clone.get_var_by_name(f'x_{feature_index}') for feature_index in range(number_features)],
        'output': [mdl_clone.get_var_by_name(f'o_{output_index}') for output_index in range(number_outputs)],
        'binary': mdl_clone.binary_var_list(number_outputs - 1, name='q')
    }
    input_constraints = mdl_clone.add_constraints(
        [input_variable == feature for input_variable, feature in zip(variables['input'], network['input'])])
    mdl_clone.add_constraint(mdl_clone.sum(variables['binary']) >= 1)
    insert_tjeng_output_constraints(mdl_clone, bounds['output'], variables, network['output'])
    explication_mask = np.ones_like(network['input'], dtype=bool)
    for constraint_index, constraint in enumerate(input_constraints):
        mdl_clone.remove_constraint(constraint)
        explication_mask[constraint_index] = False
        mdl_clone.solve()
        if mdl_clone.solution is not None:
            mdl_clone.add_constraint(constraint)
            explication_mask[constraint_index] = True
    mdl_clone.end()
    return explication_mask


def get_minimal_explication(dataset_name, metrics):
    (x_train, _1), (x_val, _2), (x_test, _3) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    model = load_model(dataset_name)
    layers = model.layers
    mdl, bounds = build_network(x, layers)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    start_time = time()
    for (network_index, network_input), network_output in zip(x_test.iterrows(), y_pred):
        network = {'input': network_input, 'output': network_output}
        minimal_explication(mdl, bounds, network)
    end_time = time()
    metrics['explication_times'].append(end_time - start_time)
    mdl.end()
