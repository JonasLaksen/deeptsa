import os
import random
from datetime import datetime

import numpy as np
import pandas
import tensorflow as tf
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from src.models.bidir import BidirLSTM
from src.models.stacked_lstm import StackedLSTM
from src.pretty_print import print_for_master_thesis_compact, print_for_master_thesis
from src.utils import load_data, plot_one, predict_plots, write_to_json_file, get_features

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
    set_seed(seed)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    directory = f'{experiment_results_directory}/{model_generator.__name__}-{"-".join([str(x) for x in layer_sizes])}-{sub_experiment_timestamp}'

    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, X_stocks, scaler_y = load_data(feature_list, y_features, train_portion,
                                                                   test_portion,
                                                                   True)

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)
    meta = {
        'dropout': dropout_rate,
        'epochs': epochs,
        'time': sub_experiment_timestamp,
        'features': ', '.join(feature_list),
        'model-type': model_generator.__name__,
        'layer-sizes': f"[{', '.join(str(x) for x in layer_sizes)}]",
        'loss': loss_function,
        'seed': seed,
        'X-train-shape': list(X_train.shape),
        'X-val-shape': list(X_val.shape),
        'y-train-shape': list(y_train.shape),
        'y-val-shape': list(y_val.shape),
        'X-stocks': list(X_stocks)
    }

    model = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                            dropout=dropout_rate)
    model.compile(optimizer='adam', loss=loss_function)
    history = model.fit(X_train, y_train,
                        validation_data=([X_val, y_val]),
                        batch_size=batch_size, epochs=epochs, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=100, restore_best_weights=True)]
                        )
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('is bidir?')
    print(model_generator == BidirLSTM)

    evaluation = predict_plots(model, X_train, y_train, X_val, y_val, scaler_y, y_features[0], X_stocks,
                               directory, is_bidir=model_generator == BidirLSTM)
    plot_one('Loss history', [history.history['loss'], history.history['val_loss']], ['Training loss', 'Test loss'],
             ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    write_to_json_file(history.history, f'{directory}/loss_history.json', )
    write_to_json_file(evaluation, f'{directory}/evaluation.json')
    write_to_json_file(meta, f'{directory}/meta.json', )


price = ['price']
trading_features = ['open', 'high', 'low', 'volume', 'direction', 'change']
trading_features_with_price = ['price'] + trading_features
sentiment_features = ['positive', 'negative', 'neutral', 'positive_prop', 'negative_prop',
                      'neutral_prop']  # , ['all_positive', 'all_negative', 'all_neutral']]#, ['all_positive', 'all_negative', 'all_neutral']]
trendscore_features = ['trendscore']

# feature_subsets = [
#     trading_features_with_price,
#     sentiment_features,
#     trendscore_features,
#     trading_features_with_price + sentiment_features,
#     trading_features_with_price + trendscore_features,
#     sentiment_features + trendscore_features,
#     trading_features_with_price + sentiment_features + trendscore_features
# ]

# price = ['prev_price_0', 'prev_price_1', 'prev_price_2'] + ['price']
# trading_features = ['prev_volume_0', 'prev_volume_1', 'prev_volume_2'] + trading_features
# sentiment_features = [f'prev_{feature}_{i}' for i, feature in enumerate(['positive', 'negative','neutral'])] + sentiment_features
# trendscore_features = [f'prev_{feature}_{i}' for i, feature in enumerate(trendscore_features)] + trendscore_features

feature_subsets = [price,
                   price + trading_features,
                   price + sentiment_features,
                   price + trendscore_features,
                   price + trading_features + sentiment_features + trendscore_features
                   ]

configurations = [
    {
        'lstm_type': StackedLSTM,
        'layers': [160]
    }
    # , {
    #     'lstm_type': StackedLSTM,
    #     'layers': [80, 80]
    # }, {
    #     'lstm_type': StackedLSTM,
    #     'layers': [54, 53, 53]
    # },
    # {
    #     'lstm_type': BidirLSTM,
    #     'layers': [160]
    # }, {
    #     'lstm_type': BidirLSTM,
    #     'layers': [80, 80]
    # }, {
    #     'lstm_type': BidirLSTM,
    #     'layers': [54, 53, 53]
    # },
]

n = 100
number_of_epochs = 5000

for seed in range(3)[:n]:
    for features in feature_subsets[:n]:
        for configuration in configurations:
            experiment_hyperparameter_search(seed=seed, layer_sizes=configuration['layers'],
                                             dropout_rate=.0,
                                             loss_function='mae',
                                             epochs=number_of_epochs,
                                             y_features=['next_change'],
                                             feature_list=features,
                                             model_generator=configuration['lstm_type'])

print_folder = f'server_results/feature_search.py/2020-07-10_22.57.53/*/'
# print_for_master_thesis(print_folder, ['features', 'layer'], compact=True, fields_to_show=['features'])
# print_for_master_thesis(print_folder, ['features', 'model-type', 'layer'] )

print_for_master_thesis_compact(print_folder, ['features', 'layer', 'model-type'])
