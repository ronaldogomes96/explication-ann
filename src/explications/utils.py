import numpy as np
import pandas as pd

from ortools.linear_solver import pywraplp

from src.datasets.utils import read_all_datasets
from src.models.utils import load_model


def get_input_variables_and_bounds(solver, x):
    input_variables = []
    bounds = []
    for index, column in enumerate(x.columns):
        unique_values = x[column].unique()
        x_min, x_max = unique_values.min(), unique_values.max()
        name = f'x_{index}'
        if len(unique_values) == 2:
            input_variables.append(solver.IntVar(0, 1, name))
        elif np.any(unique_values.astype('int64') != unique_values.astype('float64')):
            input_variables.append(solver.NumVar(x_min, x_max, name))
        else:
            input_variables.append(solver.IntVar(x_min, x_max, name))
        bounds.append((x_min, x_max))
    return input_variables, bounds


def get_intermediate_variables(solver, layer_index, number_neurons):
    infinity = solver.infinity()
    intermediate_variables = []
    for neuron_index in range(number_neurons):
        intermediate_variables.append(solver.NumVar(0, infinity, f'y_{layer_index}_{neuron_index}'))
    return intermediate_variables


def get_decision_variables(solver, layer_index, number_neurons):
    decision_variables = []
    for neuron_index in range(number_neurons):
        decision_variables.append(solver.IntVar(0, 1, f'z_{layer_index}_{neuron_index}'))
    return decision_variables


def get_output_variables(solver, number_outputs):
    infinity = solver.infinity()
    output_variables = []
    for output_index in range(number_outputs):
        output_variables.append(solver.NumVar(-infinity, infinity, f'o_{output_index}'))
    return output_variables


def maximize(solver, variable):
    solver.Maximize(variable)
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception('The variable cannot be maximized')
    objective = solver.Objective().Value()
    solver.Objective().Clear()
    return objective


def minimize(solver, variable):
    solver.Minimize(variable)
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise Exception('The variable cannot be minimized')
    objective = solver.Objective().Value()
    solver.Objective().Clear()
    return objective


def build_tjeng_network(solver, layers, variables):
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
            upper_bound = maximize(solver, result)
            if upper_bound <= 0 and layer != last_layer:
                solver.Add(y == 0, f'c_{layer_index}_{neuron_index}')
                continue
            lower_bound = minimize(solver, result)
            if lower_bound >= 0 and layer != last_layer:
                solver.Add(y == result, f'c_{layer_index}_{neuron_index}')
                continue
            if layer != last_layer:
                solver.Add(y <= result - lower_bound * (1 - z))
                solver.Add(y >= result)
                solver.Add(y <= upper_bound * z)
            else:
                solver.Add(y == result)
                output_bounds.append((lower_bound, upper_bound))
    return solver, output_bounds


def build_network(x, layers):
    solver = pywraplp.Solver.CreateSolver('SAT')
    variables = {'decision': [], 'intermediate': []}
    bounds = {}
    variables['input'], bounds['input'] = get_input_variables_and_bounds(solver, x)
    last_layer = layers[-1]
    for layer_index, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]
        if layer == last_layer:
            variables['output'] = get_output_variables(solver, number_variables)
            break
        variables['intermediate'].append(get_intermediate_variables(solver, layer_index, number_variables))
        variables['decision'].append(get_decision_variables(solver, layer_index, number_variables))
    solver, bounds['output'] = build_tjeng_network(solver, layers, variables)
    return solver, bounds


def get_minimal_explication(dataset_name):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    layers = load_model(dataset_name).layers
    solver, bounds = build_network(x, layers)
    print(solver, bounds)
