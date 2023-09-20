import numpy as np
import pandas as pd

from docplex.mp.model import Model

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


def build_network(x, layers):
    mdl = Model()
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


def get_minimal_explication(dataset_name):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    layers = load_model(dataset_name).layers
    mdl, bounds = build_network(x, layers)
    print(mdl, bounds)
