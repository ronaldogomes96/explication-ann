import logging
import numpy as np

from pathlib import Path

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.explications.utils import get_minimal_explication
from src.models.utils import eval, train, is_model_trained

if __name__ == '__main__':
    Path('log').mkdir(exist_ok=True)
    dataset_name = 'iris'
    logging.basicConfig(
        level=logging.INFO,
        filename=f'log/{dataset_name}.log',
        filemode='w',
        encoding='utf-8',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if not is_dataset_prepared(dataset_name):
        prepare_and_save_dataset(dataset_name)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    if not is_model_trained(dataset_name):
        train(dataset_name, x_train, y_train, x_val, y_val)
    eval(dataset_name, x_test, y_test)
    metrics = {
        'explication_times': []
    }
    for i in range(10):
        get_minimal_explication(dataset_name, metrics, use_box=True)
    avg_explication_time = np.average(metrics['explication_times'])
    logging.info(f'Average time of explication: {avg_explication_time:.2f} seconds.')
