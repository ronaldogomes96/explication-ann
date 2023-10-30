import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from dotenv import load_dotenv

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.explications.utils import build_network, minimal_explications
from src.metrics.utils import create_metrics, log_metrics, prepare_metrics
from src.models.utils import evaluate, is_model_trained, load_model, train


def load_datasets_from_env():
    def fn(dataset):
        if ':' in dataset:
            name, limit = dataset.split(':')
            return {'name': name, 'limit': int(limit)}
        return {'name': dataset}
    datasets = os.getenv('DATASETS').split(',')
    return tuple(map(fn, datasets))


if __name__ == '__main__':
    Path('log').mkdir(exist_ok=True)
    load_dotenv()
    datasets = load_datasets_from_env()
    percentage_progress = 0
    for dataset_index, dataset in enumerate(datasets):
        tf.keras.backend.clear_session()
        dataset_name = dataset['name']
        logging.basicConfig(
            level=logging.INFO,
            filename=f'log/{dataset_name}.log',
            filemode='w',
            encoding='utf-8',
            format='[ %(asctime)s - %(levelname)s ] %(message)s',
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
        metrics = create_metrics(dataset_name)
        if 'limit' in dataset:
            x_test = x_test.head(dataset['limit'])
        y_pred = np.argmax(model.predict(x_test), axis=1)
        mdl, bounds = build_network(x, layers, metrics)
        number_executions = int(os.getenv('EXECUTIONS'))
        step = 1 / (2 * number_executions * len(datasets))
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITHOUT BOX')
        for execution in range(number_executions):
            print(f'{percentage_progress * 100:.2f}%')
            log_output = not execution
            minimal_explications(mdl, bounds, layers, x_test, y_pred, metrics, log_output)
            percentage_progress += step
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITH BOX')
        for execution in range(number_executions):
            print(f'{percentage_progress * 100:.2f}%')
            log_output = not execution
            minimal_explications(mdl, bounds, layers, x_test, y_pred, metrics, log_output, use_box=True)
            percentage_progress += step
        mdl.end()
        final_metrics = prepare_metrics(metrics, number_executions, len(x_test))
        log_metrics(final_metrics)
        percentage_progress = (dataset_index + 1) / len(datasets)
    print(f'{percentage_progress * 100:.2f}%')
