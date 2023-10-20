import re
import os

from dotenv import load_dotenv
from os.path import dirname, join


def load_datasets_from_env():
    datasets = os.getenv('DATASETS').split(',')
    return tuple(map(lambda dataset: dataset.split(':')[0] if ':' in dataset else dataset, datasets))


def extract_explication(line):
    regex = r'- Relevant: (\[.+\])'
    result = re.search(regex, line)
    return result.group(1)


if __name__ == '__main__':
    load_dotenv()
    dataset_names = load_datasets_from_env()
    for dataset_name in dataset_names:
        log_path = join(dirname(__file__), 'log', f'{dataset_name}.log')
        with open(log_path) as f:
            lines = list(filter(lambda line: '- Relevant:' in line, f.readlines()))
            explications = list(map(extract_explication, lines))
        result = 'equal'
        half = int(len(explications) / 2)
        for i in range(half):
            if explications[i] != explications[half + i]:
                result = 'not equal'
                break
        print(f'Explications of dataset {dataset_name} are {result}.')
