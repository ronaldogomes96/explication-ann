import numpy as np

from docplex.mp.model import Model


def get_input_variables_and_bounds(mdl: Model, x, metrics):
    input_variables = []
    input_bounds = []
    for column_index, column in enumerate(x.columns):
        unique_values = x[column].unique()
        lower_bound, upper_bound = unique_values.min(), unique_values.max()
        name = f'x_{column_index}'
        if len(unique_values) == 2:
            input_variables.append(mdl.binary_var(name=name))
            metrics['binary_vars'] += 1
        else:
            input_variables.append(mdl.continuous_var(lb=lower_bound, ub=upper_bound, name=name))
            metrics['continuous_vars'] += 1
        input_bounds.append((lower_bound, upper_bound))
    return input_variables, np.array(input_bounds)


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
