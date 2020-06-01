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

# seed = int(sys.argv[1]) if sys.argv[1] else 0
# type_search = sys.argv[2] if sys.argv[2] else 'hyper'
# layer_sizes = list(map(int, sys.argv[3].split(","))) if sys.argv[3] else [999]
# model_type = sys.argv[4] if sys.argv[4] else 'stacked'
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

layer_sizes = [128]
dropout = .1

def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


def experiment_price_multiple_steps(epochs, y_type='next_price', feature_list=[], experiment_name="default_multiple_steps"):
    experiment_timestamp = datetime.now()
    description = 'Analyse all stocks with multiple steps backwards and evaluate'
    X_train, y_train, X_val, y_val, y_dir, X_stocks, scaler_y = load_data(feature_list, y_type, .9)

    X = np.append(X_train, X_val, axis=1)
    n_features, batch_size = calculate_n_features_and_batch_size(X_train)

    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]

    lstm = LSTMOneOutput(**{
            'X_stocks': X_stocks,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_list': feature_list,
            'dropout': dropout,
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
    filename_midfix = f'{os.path.basename(__file__)}/{experiment_timestamp}'
    directory = f'results/{filename_midfix}/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = lstm.generate_general_model_results(
        scaler_y=scaler_y, y_type=y_type, title="Multiple steps back", filename=f'{ directory }/plot' )
    lstm.gen_model.save_weights(f'{directory}/weights.h5' )
    with open(
            f'{directory}/loss_history.txt',
    'a+') as f:
        f.write(str(losses['general_loss']))
        f.write(str(losses['general_val_loss']))
    with open(
            f'{directory}/evaluation.json',
            'a+') as f:
        f.write(json.dumps(evaluation, indent=4))
    with open(
            f'{directory}/meta.txt',
    'a+') as f:
        f.write(lstm.meta(description, epochs))

    plot_one('Loss history', [losses['general_loss'], losses['general_val_loss']], ['Training loss', 'Validation loss'], ['Epoch', 'Loss'])


feature_list = get_features()
feature_list = get_feature_list_lags(feature_list, lags=1)
experiment_price_multiple_steps(5000, y_type="next_price", feature_list=feature_list, experiment_name="default_multiple_steps_1")

feature_list = get_features()
feature_list = get_feature_list_lags(feature_list, lags=2)
experiment_price_multiple_steps(5000, y_type="next_price", feature_list=feature_list, experiment_name="default_multiple_steps_2")

feature_list = get_features()
feature_lags = get_feature_list_lags(["price"], lags=1)
feature_list = feature_list + feature_lags
experiment_price_multiple_steps(5000, y_type="next_price", feature_list=feature_list, experiment_name="default_multiple_price_steps_1")

feature_list = get_features()
feature_lags = get_feature_list_lags(["price"], lags=2)
feature_list = feature_list + feature_lags
experiment_price_multiple_steps(5000, y_type="next_price", feature_list=feature_list, experiment_name="default_multiple_price_steps_2")


