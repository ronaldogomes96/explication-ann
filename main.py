import logging
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def transform(x, columns):
    x = StandardScaler().fit_transform(x)
    return pd.DataFrame(x, columns=columns)


if __name__ == '__main__':
    Path('log').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename='log/app.log',
        filemode='w',
        encoding='utf-8',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    df = pd.read_csv('iris.csv')
    features, target = df.columns[:-1], df.columns[-1]
    df[target] = pd.factorize(df[target])[0]
    x, y = df.loc[:, features], df.loc[:, target]
    x = transform(x, features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    logging.info('End of execution')
