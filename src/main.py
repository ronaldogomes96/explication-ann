import logging
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time


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
    input_shape = (x_train.shape[1],)
    n_layers = 3
    n_neurons = 3
    n_epochs = 1000
    batch_size = 10
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons, input_shape=input_shape, activation='relu'))
    for _ in range(n_layers - 2):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(n_neurons, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = (tf.keras.metrics.SparseCategoricalAccuracy(),)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    patience = int(n_epochs * 0.1)
    callbacks = (
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ModelCheckpoint(filepath='iris.keras')
    )
    start_time = time()
    model.fit(
        x_train,
        y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=0
    )
    end_time = time()
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    logging.info(f'Time of training: {end_time - start_time:.2f} seconds.')
    logging.info(f'Loss: {loss}')
    logging.info(f'Accuracy: {accuracy * 100.0:.2f}%')
