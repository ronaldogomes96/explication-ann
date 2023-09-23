import logging

from pathlib import Path

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.explications.utils import get_minimal_explications
from src.models.utils import eval, train, is_model_trained


def create_metrics():
    return {
        'with_box': {
            'accumulated_time': 0,
            'average_time': 0.0
        },
        'without_box': {
            'accumulated_time': 0,
            'average_time': 0.0
        }
    }


def log_metrics(dataset_name, x_test, y_test, metrics):
    columns = x_test.columns
    logging.info(f'EXPLICATIONS FOR DATASET {dataset_name.upper()}')
    box_explications = metrics['with_box']['explications']
    explications = metrics['without_box']['explications']
    for (x_index, x), y, box_explication, explication in zip(x_test.iterrows(), y_test, box_explications, explications):
        logging.info('--------------------------------------------------------------------------------')
        logging.info(f'>>> INPUT {x_index}\n{x}')
        logging.info(f'>>> OUTPUT\n{y}')
        logging.info('>>> EXPLICATIONS WITH BOX')
        logging.info(f'- Relevant: {list(columns[box_explication["explication_mask"]])}')
        logging.info(f'- Irrelevant by box: {list(columns[box_explication["box_mask"]])}')
        logging.info('>>> EXPLICATIONS WITHOUT BOX')
        logging.info(f'- Relevant: {list(columns[explication["explication_mask"]])}')
    logging.info('--------------------------------------------------------------------------------')
    logging.info('METRICS')
    logging.info(f'- Average time with box: {metrics["with_box"]["average_time"]:.2f} seconds.')
    logging.info(f'- Average time without box: {metrics["without_box"]["average_time"]:.2f} seconds.')


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
    for i in range(number_executions):
        metrics['with_box']['explications'] = get_minimal_explications(dataset_name, metrics, use_box=True)
        metrics['without_box']['explications'] = get_minimal_explications(dataset_name, metrics)
    metrics['with_box']['average_time'] = metrics['with_box']['accumulated_time'] / number_executions
    metrics['without_box']['average_time'] = metrics['without_box']['accumulated_time'] / number_executions
    log_metrics(dataset_name, x_test, y_test, metrics)
