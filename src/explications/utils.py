import numpy as np
import logging

from docplex.mp.model import Model
from time import time

from src.explications.box import box_relax_input_to_bounds, box_has_solution
from src.explications.milp import get_input_variables_and_bounds, get_intermediate_variables, get_output_variables, \
    get_decision_variables
from src.explications.tjeng import build_tjeng_network, insert_tjeng_output_constraints
from src.explications.optimization import get_otimal_bounds, get_tjeng_variables


def build_network(x, layers, metrics):
    logging.info(f'Creating MILP model for the dataset {metrics["dataset_name"]}...')
    start_time = time()

    mdl_initial = Model(name='initial_bounds')
    #mdl = Model(name='original')

    variables = {'decision': [], 'intermediate': []}
    bounds = {}
    variables['input'], bounds['input'] = get_input_variables_and_bounds(mdl_initial, x, metrics)
    last_layer = layers[-1]

    for layer_index, layer in enumerate(layers):
        number_variables = layer.get_weights()[0].shape[1]
        metrics['continuous_vars'] += number_variables

        if layer == last_layer:
            variables['output'] = get_output_variables(mdl_initial, number_variables)
            break
        variables['intermediate'].append(get_intermediate_variables(mdl_initial, layer_index, number_variables))
        variables['decision'].append(get_decision_variables(mdl_initial, layer_index, number_variables))
        metrics['binary_vars'] += number_variables

    mdl = mdl_initial.clone(new_name='Original')
    bounds['layers'], bounds['output'] = build_tjeng_network(mdl_initial, layers, variables, metrics)

    # add metrics of insert_tjeng_output_constraints
    metrics['constraints'] += len(variables['input'])       # input constraints
    metrics['binary_vars'] += len(variables['output']) - 1  # q
    metrics['constraints'] += len(variables['output'])      # sum of q variables and q constraints

    ## Add aqui meus logs de contraints mudadas apos o box
    logging.info('Number of variables and constraints:'
                 f'\n- Binary variables: {metrics["binary_vars"]}'
                 f'\n- Continuous variables: {metrics["continuous_vars"]}'
                 f'\n- Constraints: {metrics["constraints"]}')
    logging.info(f'Time of MILP model creation: {time() - start_time:.4f} seconds.')

    return mdl, bounds


def prepare_explication(features, explication_mask, box_mask):
    return {
        'relevant': list(features[explication_mask]),
        'irrelevant': list(features[~explication_mask]),
        'irrelevant_by_box': list(features[box_mask]),
        'irrelevant_by_solver': list(features[np.bitwise_xor(box_mask, ~explication_mask)])
    }


def log_explication(explication):
    logging.info('EXPLICATION:'
                 f'\n- Length of explication: {len(explication["relevant"])}'
                 f'\n- Relevant: {explication["relevant"]}'
                 f'\n- Irrelevant: {explication["irrelevant"]}')
    if explication['irrelevant_by_box']:
        logging.info('BOX EXPLICATION:'
                     f'\n- Irrelevant by box: {explication["irrelevant_by_box"]}'
                     f'\n- Irrelevant by solver: {explication["irrelevant_by_solver"]}')


def minimal_explication(mdl: Model, layers, bounds, network, metrics, log_output, use_box, use_box_optimization):
    if log_output:
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'INPUT:\n{network["input"]}')
        logging.info(f'OUTPUT:\n{network["output"]}')

    mdl = mdl.clone(new_name='clone')

    number_features = len(bounds['input'])
    number_outputs = len(bounds['output'])

    variables = {
        'input': [mdl.get_var_by_name(f'x_{feature_index}') for feature_index in range(number_features)],
        'output': [mdl.get_var_by_name(f'o_{output_index}') for output_index in range(number_outputs)],
        'binary': mdl.binary_var_list(number_outputs - 1, name='q')
    }

    input_constraints = mdl.add_constraints(
        [input_variable == feature for input_variable, feature in zip(variables['input'], network['input'])])
    mdl.add_constraint(mdl.sum(variables['binary']) >= 1)
    insert_tjeng_output_constraints(mdl, bounds['output'], variables, network['output'])

    explication_mask = np.ones_like(network['input'], dtype=bool)
    box_mask = np.zeros_like(network['input'], dtype=bool)

    for constraint_index, constraint in enumerate(input_constraints):
        mdl.remove_constraint(constraint)
        mdl_box = mdl.clone(new_name='clone_box')

        explication_mask[constraint_index] = False
        if use_box:
            ## Tempo de explicação do box
            start_time = time()
            relaxed_input_bounds = box_relax_input_to_bounds(network['input'], bounds['input'], ~explication_mask)

            has_solution, box_bounds = box_has_solution(relaxed_input_bounds, layers, network['output'])

            metrics['accumulated_box_time'] += (time() - start_time)

            if has_solution:
                box_mask[constraint_index] = True
                continue
            elif use_box_optimization:
                otimal_bounds = get_otimal_bounds(bounds, box_bounds)
                tjeng_variables = get_tjeng_variables(mdl_box, bounds, layers)
                build_tjeng_network(mdl_box, layers, tjeng_variables, metrics, otimal_bounds)

        if mdl_box.solve(log_output=False) is not None:
            mdl.add_constraint(constraint)
            explication_mask[constraint_index] = True
    mdl.end()
    explication = prepare_explication(network['features'], explication_mask, box_mask)
    if use_box:
        metrics['irrelevant_by_box'] += len(explication['irrelevant_by_box'])
        metrics['irrelevant_by_solver'] += len(explication['irrelevant_by_solver'])
    return explication


def minimal_explications(mdl: Model, bounds, layers, x_test, y_pred, metrics,
                         log_output=False, use_box=False, use_box_optimization=False):
    features = x_test.columns
    key = 'accumulated_time_with_box' if use_box else 'accumulated_time_without_box'
    start_time = time()

    for (network_index, network_input), network_output in zip(x_test.iterrows(), y_pred):
        network = {'input': network_input,
                   'output': network_output,
                   'features': features}
        explication = minimal_explication(mdl, layers, bounds, network, metrics,
                                          log_output, use_box, use_box_optimization)

        log_output and log_explication(explication)
    metrics[key] += (time() - start_time)
