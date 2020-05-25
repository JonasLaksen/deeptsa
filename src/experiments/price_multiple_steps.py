import json
import os
import pathlib
import random
import sys
from datetime import datetime
from functools import reduce

import numpy as np
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, get_features, plot_one, get_feature_list_lags

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


def experiment_price_multiple_steps(epochs, n_stocks=100, y_type='next_price', feature_list=[] ):
    experiment_timestamp = datetime.now()
    # description = 'Gå gjennom alle aksjer med priser bak i tid som features'
    X, y, y_dir, X_stocks, scaler_y = load_data(feature_list, y_type)
    X, y, y_dir, X_stocks = X, y, y_dir, X_stocks
    training_size = int(.9 * len(X[0]))

    X_train, y_train = X[:, :training_size], y[:, :training_size]
    X_val, y_val = X[:, training_size:], y[:, training_size:]

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)


    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]



    lstm = LSTMOneOutput(**{
            'X_stocks': X_stocks,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_list': feature_list,
            'dropout': .1,
            'optimizer': tf.keras.optimizers.Adam(.001),
            'loss': 'MSE',
            'model_generator': StackedLSTM,
            'layer_sizes': layer_sizes,
            'seed': seed,
            'n_features': n_features,
            'batch_size': batch_size,
            'stock_list': stock_list
        })

    losses = lstm.train(
        gen_epochs=epochs,
        spech_epochs=0,
        copy_weights_from_gen_to_spec=False,
        load_spec=False,
        load_gen=False,
        train_general=True,
        train_specialized=False)
    evaluation = lstm.generate_general_model_results(
        scaler_y=scaler_y, y_type=y_type
    )



feature_list = get_features(trading=False)
# feature_lags = get_feature_list_lags(["price"], lags=2);
# feature_list = feature_list + feature_lags
# feature_list = [feature for feature in feature_list if feature != 'price']
print(feature_list)
experiment_price_multiple_steps(10, n_stocks=43, feature_list=feature_list)
