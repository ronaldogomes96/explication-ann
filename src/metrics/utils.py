import logging


def create_metrics(dataset_name):
    return {
        'dataset_name': dataset_name,
        'accumulated_time_with_box': 0,
        'accumulated_time_without_box': 0,
        'accumulated_box_time': 0,
        'irrelevant_by_box': 0,
        'continuous_vars': 0,
        'binary_vars': 0,
        'constraints': 0
    }


def prepare_metrics(metrics, number_executions, number_features, len_x):
    number_explications = number_executions * len_x
    irrelevant_by_box = metrics['irrelevant_by_box'] / number_executions
    calls_to_solver = number_features * len_x - irrelevant_by_box
    percentage_irrelevant_by_box = metrics['irrelevant_by_box'] / (number_explications * number_features)
    percentage_calls_to_solver = 1 - percentage_irrelevant_by_box
    return {
        'avg_time_with_box': metrics['accumulated_time_with_box'] / number_explications,
        'avg_time_without_box': metrics['accumulated_time_without_box'] / number_explications,
        'avg_time_box': metrics['accumulated_box_time'] / number_explications,
        'irrelevant_by_box': irrelevant_by_box,
        'calls_to_solver': calls_to_solver,
        'percentage_irrelevant_by_box': percentage_irrelevant_by_box,
        'percentage_calls_to_solver': percentage_calls_to_solver,
        'binary_vars': metrics['binary_vars'],
        'continuous_vars': metrics['continuous_vars'],
        'constraints': metrics['constraints']
    }


def log_metrics(metrics):
    avg_time_with_box = metrics['avg_time_with_box']
    avg_time_without_box = metrics['avg_time_without_box']
    logging.info('--------------------------------------------------------------------------------')
    logging.info('METRICS PER EXPLICATION')
    logging.info(f'- Average time with box: {avg_time_with_box:.4f} seconds')
    logging.info(f'  > Average time spent on box: {metrics["avg_time_box"]:.4f} seconds')
    logging.info(f'  > Irrelevant by box: {int(metrics["irrelevant_by_box"])} '
                 f'({metrics["percentage_irrelevant_by_box"] * 100:.2f}%)')
    logging.info(f'  > Calls to solver: {int(metrics["calls_to_solver"])} '
                 f'({metrics["percentage_calls_to_solver"] * 100:.2f}%)')
    logging.info(f'- Average time without box: {avg_time_without_box:.4f} seconds')
    logging.info('NUMBER OF VARIABLES AND CONSTRAINTS')
    logging.info(f'- Number of binary variables: {metrics["binary_vars"]}')
    logging.info(f'- Number of continuous variables: {metrics["continuous_vars"]}')
    logging.info(f'- Number of constraints: {metrics["constraints"]}')
    logging.info('FINAL RESULT')
    result, diff = ('better', avg_time_without_box - avg_time_with_box) if avg_time_with_box < avg_time_without_box \
        else ('worse', avg_time_with_box - avg_time_without_box)
    logging.info(f'- Running with box was {result} than without box {diff:.4f} seconds')
