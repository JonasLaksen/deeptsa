import json
import os
import random
import numpy as np
import pandas
import tensorflow as tf

from datetime import datetime
from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.pretty_print import print_for_master_thesis_compact
from src.utils import load_data, get_features, plot_one, plot, evaluate, predict_plots
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

def experiment_hyperparameter_search(seed, layer, dropout_rate, loss_function, epochs, y_type, feature_list):
    set_seed(seed)
    print(feature_list)
    sub_experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    description = 'Hyperparameter søk'
    train_portion, validation_portion, test_portion = .8, .1, .1
    X_train, y_train, X_val, y_val, _, X_stocks, scaler_y = load_data(feature_list, y_type, train_portion, test_portion, True)
    X = np.append(X_train, X_val, axis=1)
    y = np.append(y_train, y_val, axis=1)
    training_size = X_train.shape[1]
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
        'optimizer': 'adam' ,
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
    directory = f'{experiment_results_directory}/{sub_experiment_timestamp}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(lstm.gen_model, X_train, y_train, X_val, y_val, scaler_y, y_type, X_stocks, directory)
    scores = lstm.gen_model.evaluate(X_val, y_val, batch_size=batch_size)
    print(scores)
    plot_one('Loss history', [losses['general_loss'], losses['general_val_loss']], ['Training loss', 'Test loss'], ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    with open(
            f'{directory}/loss_history.json',
            'a+') as f:
        f.write(json.dumps(losses, indent=4))
    with open(
            f'{directory}/evaluation.json',
            'a+') as f:
        f.write(json.dumps(evaluation, indent=4));
    with open(f'{directory}/meta.json',
              'a+') as f:
        meta = lstm.meta(description, epochs)
        f.write(json.dumps(meta, indent=4))

feature_list = get_features()
# feature_list = ['price', 'positive']
layers = [[160], [128], [32]]
dropout_rates = [.5, .2, 0]
loss_functions = ['mse', 'mae']

n = 0
number_of_epochs = 5000
for seed in range(3)[:n]:
    for layer in layers[:n]:
        for dropout_rate in dropout_rates[:n]:
            for loss_function in loss_functions[:n]:
                experiment_hyperparameter_search(seed, layer, dropout_rate, loss_function, number_of_epochs, 'next_price', feature_list)

print_folder = f'server_results/{os.path.basename(__file__)}/2020-06-18_20.09.57/*/'
# print_for_master_thesis(print_folder, ['dropout', 'layer', 'loss'], ['mean_da_rank'])
print_for_master_thesis_compact(print_folder, ['dropout', 'layer', 'loss'], fields_to_show=['dropout', 'layer', 'loss'], show_model=False)
