import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.explications.utils import build_network, minimal_explications
from src.models.utils import evaluate, is_model_trained, load_model, train


def create_metrics():
    return {
        'with_box': {
            'accumulated_time': 0,
            'accumulated_box_time': 0,
            'calls_to_box': 0
        },
        'without_box': {
            'accumulated_time': 0
        },
        'continuous_vars': 0,
        'binary_vars': 0,
        'constraints': 0
    }


if __name__ == '__main__':
    Path('log').mkdir(exist_ok=True)
    datasets = [
        {'name': 'digits'},
        {'name': 'iris'},
        {'name': 'mnist', 'limit': 10},
        {'name': 'sonar'},
        {'name': 'wine'},
    ]
    for dataset in datasets:
        tf.keras.backend.clear_session()
        dataset_name = dataset['name']
        logging.basicConfig(
            level=logging.INFO,
            filename=f'log/{dataset_name}.log',
            filemode='w',
            encoding='utf-8',
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )
        if not is_dataset_prepared(dataset_name):
            prepare_and_save_dataset(dataset_name)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
        model = load_model(dataset_name) if is_model_trained(dataset_name) \
            else train(dataset_name, x_train, y_train, x_val, y_val)
        evaluate(model, x_test, y_test)
        x = pd.concat((x_train, x_val, x_test), ignore_index=True)
        layers = model.layers
        metrics = create_metrics()
        if 'limit' in dataset:
            x_test = x_test.head(dataset['limit'])
        y_pred = np.argmax(model.predict(x_test), axis=1)
        mdl, bounds = build_network(dataset_name, x, layers, metrics)
        number_executions = 5
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITH BOX')
        for execution in range(number_executions):
            log_output = not execution
            minimal_explications(mdl, bounds, layers, x_test, y_pred, metrics, log_output, use_box=True)
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITHOUT BOX')
        for execution in range(number_executions):
            log_output = not execution
            minimal_explications(mdl, bounds, layers, x_test, y_pred, metrics, log_output)
        mdl.end()
        average_time_with_box = metrics['with_box']['accumulated_time'] / number_executions
        average_time_without_box = metrics['without_box']['accumulated_time'] / number_executions
        average_box_time = metrics['with_box']['accumulated_box_time'] / number_executions
        percentage_calls_to_box = metrics['with_box']['calls_to_box'] / (x_test.size * number_executions)
        percentage_calls_to_solver = 1 - percentage_calls_to_box
        logging.info('--------------------------------------------------------------------------------')
        logging.info('METRICS')
        logging.info(f'Average time with box: {average_time_with_box:.4f} seconds.')
        logging.info(f'> Average time spent on box: {average_box_time:.4f} seconds')
        logging.info(f'> Calls to box: {percentage_calls_to_box * 100:.2f}%')
        logging.info(f'> Calls to solver: {percentage_calls_to_solver * 100:.2f}%')
        logging.info(f'Average time without box: {average_time_without_box:.4f} seconds.')
        logging.info('COUNTERS')
        logging.info(f'Number of binary variables: {metrics["binary_vars"]}.')
        logging.info(f'Number of continuous variables: {metrics["continuous_vars"]}.')
        logging.info(f'Number of constraints: {metrics["constraints"]}.')
