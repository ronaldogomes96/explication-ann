import pandas as pd

from os.path import dirname, join


def load_dataset(transform_x_fn=None):
    dataset_path = join(dirname(__file__), 'sonar.csv')
    df = pd.read_csv(dataset_path)
    features, target = df.columns[:-1], df.columns[-1]
    x, y = df.loc[:, features], df.loc[:, target]
    if transform_x_fn:
        x = transform_x_fn(x, features)
    return x, y
