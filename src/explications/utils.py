import numpy as np
import pandas as pd
import logging

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


def minimal_explication(mdl: Model, layers, bounds, network, metrics, log_output, use_box):
    if log_output:
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'>>> INPUT\n{network["input"]}')
        logging.info(f'>>> OUTPUT\n{network["output"]}')
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
            start_time = time()
            relax_input_mask = ~explication_mask
            relaxed_input_bounds = box_relax_input_to_bounds(network['input'], bounds['input'], relax_input_mask)
            has_solution = box_has_solution(relaxed_input_bounds, layers, network['output'])
            metrics['with_box']['accumulated_box_time'] += (time() - start_time)
            if has_solution:
                box_mask[constraint_index] = True
                metrics['with_box']['calls_to_box'] += 1
                continue
        mdl_clone.solve(log_output=False)
        if mdl_clone.solution is not None:
            mdl_clone.add_constraint(constraint)
            explication_mask[constraint_index] = True
    mdl_clone.end()
    if log_output:
        logging.info('>>> EXPLICATION')
        logging.info(f'- Relevant: {list(network["features"][explication_mask])}')
        logging.info(f'- Irrelevant: {list(network["features"][~explication_mask])}')
        if use_box and np.any(box_mask):
            irrelevant_by_solver = np.bitwise_xor(box_mask, ~explication_mask)
            logging.info(f'- Irrelevant by box: {list(network["features"][box_mask])}')
            logging.info(f'- Irrelevant by solver: {list(network["features"][irrelevant_by_solver])}')


def minimal_explications(dataset_name, metrics, log_output=False, use_box=False):
    x_train, x_val, x_test = read_all_datasets(dataset_name, ignore_y=True)
    x = pd.concat((x_train, x_val, x_test), ignore_index=True)
    features = x.columns
    model = load_model(dataset_name)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    key_box = 'with_box' if use_box else 'without_box'
    layers = model.layers
    start_time = time()
    mdl, bounds = build_network(x, layers)
    for (network_index, network_input), network_output in zip(x_test.iterrows(), y_pred):
        network = {'input': network_input, 'output': network_output, 'features': features}
        minimal_explication(mdl, layers, bounds, network, metrics, log_output, use_box)
    metrics[key_box]['accumulated_time'] += (time() - start_time)
    mdl.end()
