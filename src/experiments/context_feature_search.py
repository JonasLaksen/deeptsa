import os
import random
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.bidir_state import BidirLSTMWithState
from src.models.spec_network import SpecializedNetwork
from src.models.stacked_lstm_state import StackedLSTMWithState
from src.pretty_print import print_for_master_thesis_compact, print_for_master_thesis
from src.utils import load_data, plot_one, predict_plots, write_to_json_file

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


def experiment_hyperparameter_search(seed, layer_sizes, dropout_rate, loss_function, epochs, y_features, feature_list,
                                     model_generator):
    print(layer_sizes)
    set_seed(seed)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    description = 'Hyperparameter søk'
    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, X_stocks, scaler_y = load_data(feature_list, y_features, train_portion,
                                                                   test_portion,
                                                                   True)
    X = np.append(X_train, X_val, axis=1)
    y = np.append(y_train, y_val, axis=1)
    stock_list = np.arange(len(X)).reshape((len(X), 1, 1))

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)
    lstm = LSTMOneOutput(**{
        'X_stocks': X_stocks,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'feature_list': feature_list,
        'dropout': dropout_rate,
        'optimizer': 'adam',
        'loss': loss_function,
        'model_generator': model_generator,
        'layer_sizes': layer_sizes,
        'seed': seed,
        'n_features': n_features,
        'batch_size': batch_size,
        'stock_list': stock_list
    })
    decoder = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                              dropout=dropout_rate)

    is_bidir = isinstance(decoder, BidirLSTMWithState)
    initial_states_per_layer = 4 if is_bidir else 2
    spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(stock_list), layer_sizes=layer_sizes,
                                    decoder=decoder, n_states_per_layer=initial_states_per_layer)
    spec_model.compile(optimizer='adam', loss=loss_function)

    history = spec_model.fit([X_train, stock_list], y_train,
                             validation_data=([X_val, stock_list], y_val),
                             batch_size=batch_size, epochs=epochs, shuffle=False,
                             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                         patience=100, restore_best_weights=True)]
                             )

    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(spec_model, X_train, y_train, X_val, y_val, scaler_y, y_features[0], X_stocks,
                               directory, [stock_list], is_bidir=is_bidir)
    meta = lstm.meta(description, epochs)
    plot_one('Loss history', [history.history['loss'], history.history['val_loss']], ['Training loss', 'Test loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    write_to_json_file(history.history, f'{directory}/loss_history.json', )
    write_to_json_file(evaluation, f'{directory}/evaluation.json')
    write_to_json_file(meta, f'{directory}/meta.json', )


trading_features = ['open', 'high', 'low', 'volume', 'direction', 'change']
sentiment_features = ['positive', 'negative', 'neutral', 'positive_prop', 'negative_prop',
                      'neutral_prop']  # , ['all_positive', 'all_negative', 'all_neutral']]#, ['all_positive', 'all_negative', 'all_neutral']]
trendscore_features = ['trendscore']

feature_subsets = [['price'],
                   ['price'] + trading_features,
                   ['price'] + sentiment_features,
                   ['price'] + trendscore_features,
                   ['price'] + trading_features + sentiment_features + trendscore_features
                   ]

configurations = [
    {
        'lstm_type': StackedLSTMWithState,
        'layers': [160]
    },
    # {
    #     'lstm_type': StackedLSTMWithState,
    #     'layers': [80, 80]
    # }, {
    #     'lstm_type': StackedLSTMWithState,
    #     'layers': [54, 54, 54]
    # }
]

n = 0
number_of_epochs = 5000

for seed in range(3)[:n]:
    for features in feature_subsets[:n]:
        for configuration in configurations:
            experiment_hyperparameter_search(seed=seed, layer_sizes=configuration['layers'],
                                             dropout_rate=0,
                                             loss_function='mae',
                                             epochs=number_of_epochs,
                                             y_features=['next_price'],
                                             feature_list=features,
                                             model_generator=configuration['lstm_type'])

print_folder = f'server_results/context_feature_search.py/2020-07-06_16.44.57/*/'
print_for_master_thesis(print_folder, ['features', 'layer'])
# print_for_master_thesis_compact(print_folder, ['features', 'layer'])
