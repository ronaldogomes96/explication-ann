import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def transform(x, columns):
    x = StandardScaler().fit_transform(x)
    return pd.DataFrame(x, columns=columns)


if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    features, target = df.columns[:-1], df.columns[-1]
    df[target] = pd.factorize(df[target])[0]
    x, y = df.loc[:, features], df.loc[:, target]
    x = transform(x, features)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0, stratify=y)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=0, stratify=y_test)
    print(x_train)
    print(y_train)
    print(x_val)
    print(y_val)
    print(x_test)
    print(y_test)
