import logging
import tensorflow as tf

from os.path import dirname, join
from time import time

from src.models.iris import iris


def _get_model_path(*paths):
    return join(dirname(__file__), *paths)


def train(dataset_name, x_train, y_train, x_val, y_val, x_test, y_test):
    input_shape = (x_train.shape[1],)
    params = iris.get_params()
    n_layers = params['n_layers']
    n_neurons = params['n_neurons']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_neurons, input_shape=input_shape, activation=tf.keras.activations.relu))
    for _ in range(n_layers - 2):
        model.add(tf.keras.layers.Dense(n_neurons, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(n_neurons, activation=tf.keras.activations.softmax))
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = (tf.keras.metrics.SparseCategoricalAccuracy(),)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    patience = int(n_epochs * 0.1)
    filepath = _get_model_path(dataset_name, f'{dataset_name}.keras')
    callbacks = (
        tf.keras.callbacks.EarlyStopping(patience=patience),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath)
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
