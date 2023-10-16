import re

from os.path import dirname, join


def extract_explication(line):
    regex = r'\[ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO \] - Relevant: (\[.+\])'
    result = re.search(regex, line)
    return result.group(1)


if __name__ == '__main__':
    dataset_names = ('digits', 'iris', 'mnist', 'sonar', 'wine')
    for dataset_name in dataset_names:
        log_path = join(dirname(__file__), 'log', f'{dataset_name}.log')
        with open(log_path) as f:
            lines = list(filter(lambda line: '- Relevant:' in line, f.readlines()))
            explications = list(map(extract_explication, lines))
        half = int(len(explications) / 2)
        for i in range(half):
            if explications[i] != explications[half + i]:
                raise Exception(f'Explications of dataset {dataset_name} are different')
        print(f'Explications of dataset {dataset_name} are equal')
