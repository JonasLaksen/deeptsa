import json
import os
import random
import numpy as np
import pandas
import tensorflow as tf

from datetime import datetime
from src.lstm_one_output import LSTMOneOutput
from src.models.stacked_lstm import StackedLSTM
from src.utils import load_data, get_features, plot_one, plot, evaluate
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

def predict_plots(model, X_train, y_train, X_val, y_val, scaler_y, y_type, stocklist, directory ):
    X = np.concatenate((X_train, X_val), axis=1)
    y = np.concatenate((y_train, y_val), axis=1)

    n_stocks = X_train.shape[0]

    result = model.predict([X])
    results_inverse_scaled = scaler_y.inverse_transform(result.reshape(n_stocks, -1).T).T
    y_inverse_scaled = scaler_y.inverse_transform(y.reshape(n_stocks, -1).T).T
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
    np.savetxt(f'{directory}/y.txt', y_inverse_scaled.reshape(-1))
    np.savetxt(f"{directory}/result.txt", results_inverse_scaled.reshape(-1))
    return {'training': train_evaluation, 'validation': val_evaluation}


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

# feature_list = get_features()
feature_list = ['price']
layers = [[32], [128], [160]]
dropout_rates = [0, .2, .5]
loss_functions = ['mse', 'mae']

number_of_epochs = 5000
for seed in range(3):
    for layer in layers:
        for dropout_rate in dropout_rates:
            for loss_function in loss_functions:
                experiment_hyperparameter_search(seed, layer, dropout_rate, loss_function, number_of_epochs, 'next_price', feature_list)


def print_for_master_thesis(experiment_timestamp):
    subdirectories = glob(f'results/{os.path.basename(__file__)}/{experiment_timestamp}/*/')

    subexperiments = []
    for subdirectory in subdirectories:
        meta_path = f'{subdirectory}meta.json'
        with open(meta_path, 'r') as json_file:
            meta = json.load(json_file)

        evaluation_path = f'{subdirectory}evaluation.json'
        with open(evaluation_path, 'r') as json_file:
            evaluation = json.load(json_file)

        print(evaluation)
        subexperiments.append({'seed': meta['seed'],
                    'layer': meta['layer-sizes'],
                    'dropout': meta['dropout'],
                    'loss': meta['loss'],
                    'mape': evaluation['validation']['MAPE'],
                    'mae': evaluation['validation']['MAE'],
                    'mse': evaluation['validation']['MSE'],
                    'da': evaluation['validation']['DA']})

    df = pandas.DataFrame(subexperiments)
    metrics = ['mape', 'mae', 'mse', 'da']
    for metric in metrics:
        df[f'mean_{metric}'] = df.groupby([ 'layer', 'dropout', 'loss' ])[metric].transform('mean')
        df[f'mean_{metric}_rank'] = df[f'mean_{metric}'].rank(method='dense', ascending=metric != 'da')
        df[metric] = df[metric].transform(lambda x: f'{x:.4}' if x < 10000 else int(x))
        df[f'mean_{metric}'] = df[f'mean_{metric}'].transform(lambda x: f'{x:.4}' if x < 10000 else int(x))

    df['sum_ranks'] = df[[f'mean_{metric}_rank' for metric in metrics ]].sum(axis=1)
    df = df.sort_values([ 'sum_ranks', 'layer', 'dropout', 'loss', 'seed' ])
    list_of_rows = df.to_dict('records')
    list_of_groups = zip(*(iter(list_of_rows),) * 3)

    for group in list_of_groups:
        output = f'''{group[0]['dropout']},{group[0]['layer']},{group[0]['loss']}
        { group[0]['seed'] } & { ' '.join([f"{group[0][metric]} &" for metric in metrics])} \\\\
        { group[1]['seed'] } & { ' '.join([f"{group[1][metric]} &" for metric in metrics])} \\\\
        { group[2]['seed'] } & { ' '.join([f"{group[2][metric]} &" for metric in metrics])} \\\\
        \midrule
        Mean & { ' '.join([f'{group[0][f"mean_{metric}"]} &' for metric in metrics])} \\\\
        Mean Rank & { ' '.join([f'{int(group[0][f"mean_{metric}_rank"])} &' for metric in metrics])} \\\\
        Sum rank & {int(group[0]['sum_ranks'])} \\\\
        \midrule '''

        print(output)

# print_for_master_thesis('2020-06-14_14.15.55')
