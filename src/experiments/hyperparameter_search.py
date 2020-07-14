import json
import os
import random
import numpy as np
import pandas
import tensorflow as tf

from datetime import datetime
from src.models.stacked_lstm import StackedLSTM
from src.pretty_print import print_for_master_thesis_compact, print_for_master_thesis
from src.utils import load_data, get_features, plot_one, plot, evaluate, predict_plots, write_to_json_file
from glob import glob

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
pandas.set_option('display.max_columns', 500)
pandas.set_option('display.width', 1000)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(seed)

def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]


experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
experiment_results_directory = f'results/{os.path.basename(__file__)}/{experiment_timestamp}'

def experiment_hyperparameter_search(seed, layer_sizes, dropout_rate, loss_function, epochs, y_type, feature_list):
    set_seed(seed)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    description = 'Hyperparameter søk'
    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, X_stocks, scaler_y = load_data(feature_list, [ y_type ], train_portion, test_portion, True)
    X = np.append(X_train, X_val, axis=1)
    stock_list = [np.arange(len(X)).reshape((len(X), 1, 1))]

    n_features, batch_size = calculate_n_features_and_batch_size(X_train)
    batch_size = X_train.shape[0]
    meta = {
        'dropout': dropout_rate,
        'epochs': epochs,
        'time': sub_experiment_timestamp,
        'features': ', '.join(feature_list),
        'model-type': StackedLSTM.__name__,
        'layer-sizes': f"[{', '.join(str(x) for x in layer_sizes)}]",
        'loss': loss_function,
        'seed': seed,
        'X-train-shape': list(X_train.shape),
        'X-val-shape': list(X_val.shape),
        'y-train-shape': list(y_train.shape),
        'y-val-shape': list(y_val.shape),
        'X-stocks': list(X_stocks)
    }

    model = StackedLSTM(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                            dropout=dropout_rate)
    model.compile(optimizer='adam', loss=loss_function)
    history = model.fit(X_train, y_train,
                        validation_data=([X_val, y_val]),
                        batch_size=batch_size, epochs=epochs, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=100, restore_best_weights=True)]
                        )
    directory = f'{experiment_results_directory}/{sub_experiment_timestamp}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(model, X_train, y_train, X_val, y_val, scaler_y, y_type, X_stocks,
                               directory, is_bidir=False)
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
                      'neutral_prop']
trendscore_features = ['trendscore']
feature_list = price + trading_features + sentiment_features + trendscore_features

# feature_list = ['price', 'positive']
layers = [[160], [128], [32]]
dropout_rates = [.5, .2, 0]
loss_functions = ['mae', 'mse']

n = 0
number_of_epochs = 1
for seed in range(3)[:n]:
    for layer in layers[:n]:
        for dropout_rate in dropout_rates[:n]:
            for loss_function in loss_functions[:n]:
                experiment_hyperparameter_search(seed, layer, dropout_rate, loss_function, number_of_epochs, 'next_price', feature_list)

print_folder = f'server_results/hyperparameter_search.py/2020-07-13_23.53.16/*/'
# print_for_master_thesis(print_folder, ['dropout', 'layer', 'loss'] )
print_for_master_thesis_compact(print_folder, ['dropout', 'layer', 'loss'], fields_to_show=['dropout', 'layer', 'loss'], show_model=False)
