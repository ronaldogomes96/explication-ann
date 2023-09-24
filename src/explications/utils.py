import numpy as np
import pandas as pd

from docplex.mp.model import Model
from time import time

from src.datasets.utils import read_all_datasets
from src.explications.box import box_relax_input_to_bounds, box_has_solution
from src.explications.milp import get_input_variables_and_bounds, get_intermediate_variables, get_output_variables, \
    get_decision_variables
from src.explications.tjeng import build_tjeng_network, insert_tjeng_output_constraints
from src.models.utils import load_model


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


def minimal_explication(mdl: Model, layers, bounds, network, use_box):
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
    box_mask = np.zeros_like(network['input'], dtype=bool)
    for constraint_index, constraint in enumerate(input_constraints):
        mdl_clone.remove_constraint(constraint)
        explication_mask[constraint_index] = False
        if use_box:
            relax_input_mask = ~explication_mask
            relaxed_input_bounds = box_relax_input_to_bounds(network['input'], bounds['input'], relax_input_mask)
            if box_has_solution(relaxed_input_bounds, layers, network['output']):
                box_mask[constraint_index] = True
                continue
        mdl_clone.solve(log_output=False)
        if mdl_clone.solution is not None:
            mdl_clone.add_constraint(constraint)
            explication_mask[constraint_index] = True
    mdl_clone.end()
    return {
        'explication_mask': explication_mask,
        'box_mask': box_mask
    } if use_box else {
        'explication_mask': explication_mask
    }


def get_minimal_explications(dataset_name, metrics, use_box=False):
    (x_train, _1), (x_val, _2), (x_test, _3) = read_all_datasets(dataset_name)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    model = load_model(dataset_name)
    layers = model.layers
    mdl, bounds = build_network(x, layers)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    key_box = 'with_box' if use_box else 'without_box'
    explications = []
    start_time = time()
    for (network_index, network_input), network_output in zip(x_test.iterrows(), y_pred):
        network = {'input': network_input, 'output': network_output}
        explications.append(minimal_explication(mdl, layers, bounds, network, use_box))
    end_time = time()
    metrics[key_box]['accumulated_time'] += (end_time - start_time)
    mdl.end()
    return explications
