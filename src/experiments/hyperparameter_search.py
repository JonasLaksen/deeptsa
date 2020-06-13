import json
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, get_features, plot_one, plot, evaluate

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)

def reset_seed():
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


reset_seed()

def calculate_n_features_and_batch_size(X_train):
    return X_train.shape[2], X_train.shape[0]

def predict_plots(model, X_train, y_train, X_val, y_val, scaler_y, y_type, stocklist, directory ):
    X = np.concatenate((X_train, X_val), axis=1)
    y = np.concatenate((y_train, y_val), axis=1)

    n_stocks = X_train.shape[0]

    result = model.predict([X])
    results_inverse_scaled = scaler_y.inverse_transform(result.reshape(n_stocks, -1))
    y_inverse_scaled = scaler_y.inverse_transform(y.reshape(n_stocks, -1))
    training_size = X_train.shape[1]

    result_train = results_inverse_scaled[:, :training_size].reshape(n_stocks, -1)
    result_val = results_inverse_scaled[:, training_size:].reshape(n_stocks, -1)

    y_train = y_inverse_scaled[:, :training_size].reshape(n_stocks, -1)
    y_val = y_inverse_scaled[:, training_size:].reshape(n_stocks, -1)

    val_evaluation = evaluate(result_val, y_val, y_type)
    train_evaluation = evaluate(result_train, y_train, y_type)
    print('Val: ', val_evaluation)
    print('Training:', train_evaluation)
    y_axis_label = 'Change $' if y_type == 'next_change' else 'Price $'
    plot(directory, f'Training', stocklist, result_train, y_train, ['Predicted', 'True value'], ['Day', y_axis_label] )
    plot(directory, 'Validation', stocklist, result_val, y_val, ['Predicted', 'True value'], ['Day', y_axis_label])
    # np.savetxt(f'{filename}-y.txt', y_inverse_scaled.reshape(-1))
    # np.savetxt(f"{filename}-result.txt", results_inverse_scaled.reshape(-1))
    return {'training': train_evaluation, 'validation': val_evaluation}


def experiment_hyperparameter_search(experiment_timestamp, layer, dropout_rate, loss_function, epochs, n_stocks, y_type, feature_list, layer_sizes):
    reset_seed()
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
    filename_midfix = f'{os.path.basename(__file__)}/{experiment_timestamp}/{sub_experiment_timestamp}'
    directory = f'results/{filename_midfix}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    evaluation = predict_plots(lstm.gen_model, X_train, y_train, X_val, y_val, scaler_y, y_type, X_stocks, directory)
    scores = lstm.gen_model.evaluate(X_val, y_val, batch_size=batch_size)
    print(scores)
    plot_one('Loss history', [losses['general_loss'], losses['general_val_loss']], ['Training loss', 'Test loss'], ['Epoch', 'Loss'],
             f'{directory}/loss_history.png')

    with open(
            f'{directory}/loss_history.txt',
            'a+') as f:
        f.write(str(losses['general_loss']))
        f.write(str(losses['general_val_loss']))
    with open(
            f'{directory}/evaluation.json',
            'a+') as f:
        f.write(json.dumps(evaluation, indent=4));
    with open(f'{directory}/meta.txt',
              'a+') as f:
        f.write(lstm.meta(description, epochs))

feature_list = get_features()
layers = [[32], [128], [160]]
dropout_rates = [0, .2, .5]
loss_functions = ['mse', 'mae']
experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
for layer in layers:
    for dropout_rate in dropout_rates:
        for loss_function in loss_functions:
            experiment_hyperparameter_search(experiment_timestamp, layer, dropout_rate, loss_function, 5000, 1, 'next_price', feature_list, layer_sizes=[128])
