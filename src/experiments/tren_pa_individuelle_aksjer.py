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
from src.utils import load_data, get_features, plot_one

seed = int(sys.argv[1]) if sys.argv[1] else 0
type_search = sys.argv[2] if sys.argv[2] else 'hyper'
model_type = sys.argv[4] if sys.argv[4] else 'stacked'
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


def experiment_train_on_individual_stocks(epochs, n_stocks, y_type, feature_list, layer_sizes):
    print(feature_list)
    experiment_timestamp = datetime.now()
    description = 'Gå gjennom en og en aksje og noter evalueringen'
    X, y, y_dir, X_stocks, scaler_y = load_data(feature_list, y_type)
    X, y, y_dir, X_stocks = X, y, y_dir, X_stocks
    training_size = int(.9 * len(X[0]))
    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]

    all_losses = []
    all_val_losses = []

    for i in range(min(X.shape[0], n_stocks)):
        X_stock, X_train, y_train, X_val, y_val = X_stocks[i:i + 1], \
                                                  X[i:i + 1, :training_size], \
                                                  y[i:i + 1, :training_size], \
                                                  X[i:i + 1, training_size:], \
                                                  y[i:i + 1, training_size:]
        n_features, batch_size = calculate_n_features_and_batch_size(X_train)
        lstm = LSTMOneOutput(**{
            'X_stocks': X_stock,
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_list': feature_list,
            'dropout': .1,
            'optimizer': tf.keras.optimizers.Adam(.001),
            'loss': 'MAE',
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
        directory = f'results/{filename_midfix}/aksje-{i}-{X_stocks[i]}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        evaluation = lstm.generate_general_model_results(
            scaler_y=scaler_y, y_type=y_type, title=X_stock[0], filename=f'{ directory }/plot'
        )
        all_losses.append(losses['general_loss'])
        all_val_losses.append(losses['general_val_loss'])

        with open(
                f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}-{X_stocks[i]}/loss_history.txt',
                'a+') as f:
            f.write(str(losses['general_loss']))
            f.write(str(losses['general_val_loss']))
        with open(
                f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}-{X_stocks[i]}/evaluation.json',
                'a+') as f:
            f.write(json.dumps(evaluation, indent=4));
        with open(f'results/{os.path.basename(__file__)}/{experiment_timestamp}/aksje-{i}-{X_stocks[i]}/meta.txt',
                  'a+') as f:
            f.write(lstm.meta(description, epochs))
    print(all_losses)
    np_all_losses = np.array(all_losses)
    np_all_val_losses = np.array(all_val_losses)
    means = np.mean(np_all_losses, axis=0)
    val_means = np.mean(np_all_val_losses, axis=0)
    plot_one('Loss history', [means, val_means], ['Training loss', 'Validation loss'], ['Epoch', 'Loss'])


def average_evaluation(filename):
    fileprefix = 'results/tren_pa_individuelle_aksjer.py'
    filepath = f'{fileprefix}/{filename}'
    all_folders = os.listdir(filepath)
    evaluations = []
    for folder in all_folders:
        with open(f'{filepath}/{folder}/evaluation.json') as json_file:
            evaluation = json.load(json_file)
            evaluations.append(evaluation)

    sum_training = reduce(lambda a, b: {metric: a[metric] + b[metric] for metric in a.keys()},
                          map(lambda x: x['training'], evaluations))
    avg_training = {metric: (sum_training[metric] / len(evaluations)) for metric in sum_training.keys()}
    sum_validation = reduce(lambda a, b: {metric: a[metric] + b[metric] for metric in a.keys()},
                            map(lambda x: x['validation'], evaluations))
    avg_validation = {metric: (sum_validation[metric] / len(evaluations)) for metric in sum_validation.keys()}
    with open(f'{filepath}/average.json', 'a+') as json_file:
        json_file.write(json.dumps({
            'training': avg_training,
            'validation': avg_validation
        }, indent=4))


feature_list = get_features()
sentiment_features_prop = ['positive_prop', 'negative_prop', 'neutral_prop']
sentiment_features_n = ['positive', 'negative', 'neutral']
sentiment_features_both = sentiment_features_n + sentiment_features_prop
prev_features = [f'prev_{feature}_{step}' for feature in [ "change", "price", "positive", "negative", "neutral", "trendscore", "volume" ] for step in range(3)]
experiment_train_on_individual_stocks(500,1, 'next_change', ['price'], layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=True, sentiment=False, trendscore=False), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=False, sentiment=True, trendscore=False), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=False, sentiment=False, trendscore=True), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=True, sentiment=True, trendscore=False), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=True, sentiment=False, trendscore=True), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=False, sentiment=True, trendscore=True), layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', get_features(trading=True, sentiment=True, trendscore=True), layer_sizes=[128])

prev_features_trading = [f'prev_{feature}_{step}' for feature in get_features(trading=True, sentiment=False, trendscore=False) for step in range(3)]
prev_features_sentiment = [f'prev_{feature}_{step}' for feature in get_features(trading=False, sentiment=True, trendscore=False) for step in range(3)]
prev_features_trend = [f'prev_{feature}_{step}' for feature in get_features(trading=False, sentiment=False, trendscore=True) for step in range(3)]
prev_features_all = [f'prev_{feature}_{step}' for feature in get_features(trading=True, sentiment=True, trendscore=True) for step in range(3)]

experiment_train_on_individual_stocks(500,1, 'next_change', feature_list + prev_features_trading, layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', feature_list + prev_features_sentiment, layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', feature_list + prev_features_trend, layer_sizes=[128])
experiment_train_on_individual_stocks(500,1, 'next_change', feature_list + prev_features_all, layer_sizes=[128])
