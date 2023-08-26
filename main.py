import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    features, target = df.columns[:-1], df.columns[-1]
    x, y = df.loc[:, features], df.loc[:, target]
    print(x)
    print(y)
