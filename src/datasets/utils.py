import logging
import pandas as pd

from os.path import dirname, exists, join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.datasets.iris import iris
from src.datasets.wine import wine


def _get_dataset_path(*paths):
    return join(dirname(__file__), *paths)


def _read_dataset(csv_path):
    df = pd.read_csv(csv_path)
    features, target = df.columns[:-1], df.columns[-1]
    return df.loc[:, features], df.loc[:, target]


def _save_dataset(x, y, csv_path):
    csv = pd.concat((x, y), axis=1)
    csv.to_csv(csv_path, index=False)


def _split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def _load_dataset_factory(dataset_name):
    if dataset_name == 'iris':
        return iris.load_dataset(transform_x_fn=_transform)
    elif dataset_name == 'wine':
        return wine.load_dataset(transform_x_fn=_transform)


def _transform(x, columns):
    x = MinMaxScaler().fit_transform(x)
    return pd.DataFrame(x, columns=columns)


def is_dataset_prepared(dataset_name):
    train_csv_path = _get_dataset_path(dataset_name, 'train.csv')
    validation_csv_path = _get_dataset_path(dataset_name, 'validation.csv')
    test_csv_path = _get_dataset_path(dataset_name, 'test.csv')
    if exists(train_csv_path) and exists(validation_csv_path) and exists(test_csv_path):
        logging.info(f'Dataset {dataset_name} is already prepared.')
        return True
    return False


def read_all_datasets(dataset_name, ignore_y=False):
    train_csv_path = _get_dataset_path(dataset_name, 'train.csv')
    validation_csv_path = _get_dataset_path(dataset_name, 'validation.csv')
    test_csv_path = _get_dataset_path(dataset_name, 'test.csv')
    x_train, y_train = _read_dataset(train_csv_path)
    x_val, y_val = _read_dataset(validation_csv_path)
    x_test, y_test = _read_dataset(test_csv_path)
    if ignore_y:
        return x_train, x_val, x_test
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def prepare_and_save_dataset(dataset_name):
    logging.info(f'Preparing the dataset {dataset_name}...')
    x, y = _load_dataset_factory(dataset_name)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _split_dataset(x, y)
    train_csv_path = _get_dataset_path(dataset_name, 'train.csv')
    validation_csv_path = _get_dataset_path(dataset_name, 'validation.csv')
    test_csv_path = _get_dataset_path(dataset_name, 'test.csv')
    _save_dataset(x_train, y_train, train_csv_path)
    _save_dataset(x_val, y_val, validation_csv_path)
    _save_dataset(x_test, y_test, test_csv_path)
    logging.info(f'Dataset {dataset_name} prepared.')
