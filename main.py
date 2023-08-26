import pandas as pd

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
    print(x)
    print(y)
