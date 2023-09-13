import numpy as np
import pandas as pd

from ortools.linear_solver import pywraplp

from src.datasets.utils import read_all_datasets
from src.models.utils import load_model


def get_input_domain_and_bounds(x):
    domain = []
    bounds = []
    for column in x.columns:
        unique_values = x[column].unique()
        if len(unique_values) == 2:
            domain.append('B')
        elif np.any(unique_values.astype('int64') != unique_values.astype('float64')):
            domain.append('C')
        else:
            domain.append('I')
        bounds.append((unique_values.min(), unique_values.max()))
    return domain, bounds


def build_network(x, layers):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    domain, bounds = get_input_domain_and_bounds(x)
    print(domain, bounds)
    return solver, None


def get_minimal_explication(dataset_name):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    layers = load_model(dataset_name).layers
    solver, bounds = build_network(x, layers)
