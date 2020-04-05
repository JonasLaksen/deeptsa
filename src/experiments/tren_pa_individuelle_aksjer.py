import os
import pathlib
import random
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, get_features

seed = int(sys.argv[1]) if sys.argv[1] else 0
type_search = sys.argv[2] if sys.argv[2] else 'hyper'
layer_sizes = list(map(int, sys.argv[3].split(","))) if sys.argv[3] else [999]
model_type = sys.argv[4] if sys.argv[4] else 'stacked'
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


def experiment_train_on_individual_stocks():
    experiment_timestamp = datetime.now()
    description = 'Gå gjennom en og en aksje og noter evalueringen'
    feature_list = get_features()
    X, y, y_dir, X_stocks, scaler_y = load_data(feature_list)
    training_size = int(.9 * len(X[0]))
    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]

    for i in range(X.shape[0]):
        X_stock, X_train, y_train, X_val, y_val = X_stocks[i:i + 1], \
                                                  X[i:i + 1, :training_size], \
                                                  y[i:i + 1, :training_size], \
                                                  X[i:i + 1, training_size:], \
                                                  y[i:i + 1, training_size:]
        n_features, batch_size = calculate_n_features_and_batch_size(X_train)
        print(X_train.shape)
        lstm = LSTMOneOutput(**{
            'X_stocks': X_stock,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_list': feature_list,
            'dropout': .0,
            'optimizer': tf.keras.optimizers.Adam(.001),
            'loss': 'MSE',
            'model_generator': StackedLSTM,
            'layer_sizes': layer_sizes,
            'seed': seed,
            'n_features': n_features,
            'batch_size': batch_size,
            'stock_list': stock_list
        })
        lstm.train(
            gen_epochs=10,
            spech_epochs=0,
            copy_weights_from_gen_to_spec=False,
            load_spec=False,
            load_gen=False,
            train_general=True,
            train_specialized=False)
        train_eval, val_eval = lstm.generate_general_model_results(
            scaler_y=scaler_y
        )
        pathlib.Path(f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}').mkdir(parents=True,
                                                                                                     exist_ok=True)
        with open(f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}/evaluation.txt', 'a+') as f:
            f.write(f'Training evaluation: {train_eval}\n')
            f.write(f'Validation evaluation: {val_eval}')
        with open(f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}/meta.txt', 'a+') as f:
            f.write(lstm.meta(description))


experiment_train_on_individual_stocks()
