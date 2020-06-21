import itertools
import json
import os
import random
from datetime import datetime
from glob import glob

import numpy as np
import pandas
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, get_features, plot_one, predict_plots, write_to_json_file, print_for_master_thesis

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_rows', 1000)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(seed)


def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
experiment_results_directory = f'results/{os.path.basename(__file__)}/{experiment_timestamp}'


def experiment_hyperparameter_search(seed, layer, dropout_rate, loss_function, epochs, y_type, feature_list):
    set_seed(seed)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{sub_experiment_timestamp}'

    description = 'Hyperparameter søk'
    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, _, X_stocks, scaler_y = load_data(feature_list, y_type, train_portion, test_portion,
                                                                      True)
    X = np.append(X_train, X_val, axis=1)
    y = np.append(y_train, y_val, axis=1)
    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)
    batch_size = X_train.shape[0]
    lstm = LSTMOneOutput(**{
        'X_stocks': X_stocks,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_list': feature_list,
        'dropout': dropout_rate,
        # 'optimizer': tf.keras.optimizers.Adam(.001),
        'optimizer': 'adam',
        'loss': loss_function,
        'model_generator': StackedLSTM,
        'layer_sizes': layer,
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
    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(lstm.gen_model, X_train, y_train, X_val, y_val, scaler_y, y_type, X_stocks, directory)
    scores = lstm.gen_model.evaluate(X_val, y_val, batch_size=batch_size)
    meta = lstm.meta(description, epochs)
    print(scores)
    plot_one('Loss history', [losses['general_loss'], losses['general_val_loss']], ['Training loss', 'Test loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    write_to_json_file(losses, f'{directory}/loss_history.json', )
    write_to_json_file(evaluation, f'{directory}/evaluation.json')
    write_to_json_file(meta, f'{directory}/meta.json', )


feature_list = get_features()
# feature_list = ['price', 'positive']
layers = [[160], [128], [32]]
dropout_rates = [.5, .2, 0]
loss_functions = ['mse', 'mae']

trading_features = [['price'], ['open', 'high', 'low', 'volume', 'direction', 'change']]
sentiment_features = [['positive', 'negative', 'neutral'], ['positive_prop', 'negative_prop',
                                                            'neutral_prop']]  # , ['all_positive', 'all_negative', 'all_neutral']]#, ['all_positive', 'all_negative', 'all_neutral']]
trendscore_features = [['trendscore']]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(iterable, r) for r in range(len(iterable) + 1))


all_features = trading_features + sentiment_features + trendscore_features
lol = list(powerset(all_features))
hehe = list(map(lambda subsets: sum(subsets, []), lol))
haha = list(filter(lambda x: len(x) != 0, hehe))

n = 1000
number_of_epochs = 5000
for seed in range(1)[:n]:
    for features in haha[:n]:
        experiment_hyperparameter_search(seed=seed, layer=[160], dropout_rate=0, loss_function='mae',
                                         epochs=number_of_epochs, y_type='next_price', feature_list=features)


# print_folder = f'server_results/{os.path.basename(__file__)}/2020-06-19_08.29.38/*/'
# print_for_master_thesis(print_folder, ['features'])
