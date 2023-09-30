import logging

from pathlib import Path

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.explications.utils import minimal_explications
from src.models.utils import eval, train, is_model_trained


def create_metrics():
    return {
        'with_box': {
            'accumulated_time': 0,
            'calls_to_box': 0
        },
        'without_box': {
            'accumulated_time': 0
        }
    }


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
    metrics = create_metrics()
    number_executions = 10
    logging.info('--------------------------------------------------------------------------------')
    logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITH BOX')
    for execution in range(number_executions):
        log_output = not execution
        minimal_explications(dataset_name, metrics, log_output, use_box=True)
    logging.info('--------------------------------------------------------------------------------')
    logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()} WITHOUT BOX')
    for execution in range(number_executions):
        log_output = not execution
        minimal_explications(dataset_name, metrics, log_output)
    average_time_with_box = metrics['with_box']['accumulated_time'] / number_executions
    average_time_without_box = metrics['without_box']['accumulated_time'] / number_executions
    percentage_calls_to_box = metrics['with_box']['calls_to_box'] / (x_test.size * number_executions)
    percentage_calls_to_solver = 1 - percentage_calls_to_box
    logging.info('--------------------------------------------------------------------------------')
    logging.info('METRICS')
    logging.info(f'- Average time with box: {average_time_with_box:.2f} seconds.')
    logging.info(f'  > Calls to box: {percentage_calls_to_box * 100:.2f}%')
    logging.info(f'  > Calls to solver: {percentage_calls_to_solver * 100:.2f}%')
    logging.info(f'- Average time without box: {average_time_without_box:.2f} seconds.')
