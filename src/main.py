import logging
import tensorflow as tf

from pathlib import Path
from time import time

from src.datasets.utils import is_dataset_prepared, prepare_and_save_dataset, read_all_datasets
from src.models.iris import iris

if __name__ == '__main__':
    Path('log').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename='log/app.log',
        filemode='w',
        encoding='utf-8',
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    dataset_name = 'iris'
    if not is_dataset_prepared(dataset_name):
        logging.info(f'Preparing the dataset {dataset_name}...')
        prepare_and_save_dataset(dataset_name)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = read_all_datasets(dataset_name)
    input_shape = (x_train.shape[1],)
    params = iris.get_params()
    n_layers = params['n_layers']
    n_neurons = params['n_neurons']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
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
    filepath = f'{dataset_name}.keras'
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
