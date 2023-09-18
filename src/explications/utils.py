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


def build_network(x, layers):
    solver = pywraplp.Solver.CreateSolver('SAT')
    variables = {}
    bounds = {}
    variables['input'], bounds['input'] = get_input_variables_and_bounds(solver, x)
    return solver, bounds


def get_minimal_explication(dataset_name):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    layers = load_model(dataset_name).layers
    solver, bounds = build_network(x, layers)
    print(solver, bounds)
