import copy
import csv
import json
from base64 import b64encode, b64decode
from zlib import compress, decompress
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from matplotlib import pyplot

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler



def expand(x): return np.expand_dims(x, axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(result, y):
    result = np.asarray(result).reshape((result.shape[0], -1))
    y = np.asarray(y).reshape((result.shape[0], -1))
    mape = mean_absolute_percentage_error(y, result)
    mae = mean_absolute_error(y, result)
    mse = mean_squared_error(y, result)
    accuracy_direction = mean_direction_eval(result, y)
    return {'MAPE': mape, 'MAE': mae, 'MSE': mse, 'DA': accuracy_direction}

def plot(title, stocklist, result, y):
    for i in range(len(result)):
        fig = pyplot.figure()
        fig.suptitle(f'{title}: {stocklist[i]}')
        pyplot.plot(result[i], label='Predicted')
        pyplot.plot(y[i], label='True value')
        pyplot.legend(loc='upper left')

    pyplot.show()

def direction_value(x, y):
    if x > y:
        return [0]
    else:
        return [1]

def mean_direction_eval(result, y):
    return np.mean(np.array(list(map(lambda x: direction_eval(x[0], x[1]), zip(result, y)))))

def direction_eval(result, y):
    result_pair = list(map(lambda x, y: direction_value(x, y), result[:-1], result[1:]))
    y_pair = list(map(lambda x, y: direction_value(x, y), y[:-1], y[1:]))
    return accuracy_score(y_pair, result_pair)


def make_train_val_test_set(data, training_prop=.8, validation_prop=.1, test_prop=.1):
    data_size = min(map(lambda data_group: len(data_group), data))
    train_size, val_size, test_size = int(data_size * training_prop), int(data_size * validation_prop), int(
        data_size * test_prop)
    data_train = data[:, -data_size:-data_size + train_size]
    data_val = data[:, -data_size:-data_size + train_size + val_size]
    data_test = data[:, -test_size:]

    return data_train, data_val, data_test

def create_direction_arrays(X_train, X_val, X_test, y_train, y_val, y_test):
    X = np.append(X_train, X_val, axis=1)
    y = np.append(y_train, y_val, axis=1)
    y_dir = []
    for i in range(X_train.shape[0]):
        y_dir_partial = list(map(lambda x, y: direction_value(x, y), y[i][:-1], y[i][1:]))
        y_dir.append(y_dir_partial)

    X = X[:, :-1]
    y = y[:, :-1]
    y_dir = np.array(y_dir)

    return make_train_val_test_set(X), make_train_val_test_set(y), make_train_val_test_set(y_dir)




def group_by_stock(data, training_prop=.8, validation_prop=.1, test_prop=.1):
    assert training_prop + validation_prop + test_prop == 1.
    group_by_dict = {}
    for row in data:
        try:
            group_by_dict[row[0]].append(row[1:])
        except:
            group_by_dict[row[0]] = [row[1:]]

    group_by_dict = {k: v for k, v in group_by_dict.items() if len(v) > 1600}

    data_size = len(min(group_by_dict.values(), key=len))
    train_size, val_size, test_size = int(data_size * training_prop), int(data_size * validation_prop), int(
        data_size * test_prop)
    train_data = list(
        map(lambda x: np.array(group_by_dict[x])[-data_size: -data_size + train_size], group_by_dict.keys()))
    val_data = list(
        map(lambda x: np.array(group_by_dict[x])[-data_size + train_size: -data_size + train_size + val_size],
            group_by_dict.keys()))
    test_data = list(map(lambda x: np.array(group_by_dict[x])[-test_size:], group_by_dict.keys()))
    return np.array(train_data), np.array(val_data), np.array(test_data)


def get_feature_list_lags(features, lags=0):
    all_features = copy.deepcopy(features)

    for feature in features:
        for i in range(lags):
            all_features = all_features + ['prev_' + feature + '_' + str(i)]

    return all_features


# dict: { data1: [1,2,3], data2: [4,5,6]}
def write_to_csv(filename, dict):
    try:
        with open(filename, 'w') as file:
            csv.writer(file).writerows(list(map(lambda x: [x[0]] + x[1], dict.items())))
    except:
        print(filename)


def from_args_to_filename(args):
    test = json.dumps(args)
    compressed = compress(test.encode('utf-8'), 9)
    return b64encode(compressed).decode('utf-8').replace('/', '$')


def from_filename_to_args(filename):
    decoded = b64decode(filename.replace('$', '/').split('.csv')[0])
    return decompress(decoded)


def plot_data(i, filepath):
    with open(filepath) as file:
        reader = csv.reader(file)
        fig = plt.figure(i)
        for row in reader:
            plt.plot([float(i) for i in row[1:]], label=row[0])
            plt.legend(loc='upper left')
        fig.suptitle(f'{i}')
        plt.show()

def load_data(feature_list):
    data = pd.read_csv('dataset_v2.csv', index_col=0)
    data = data.dropna()
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = data['stock'].values.reshape(-1, 1)

    try:
        scaled_X = scaler_X.fit_transform(data[[x for x in feature_list if x is not 'trendscore']].values)
        X = np.append(X, scaled_X, axis=1)
    except:
        # If there are no features to be scaled an error is thrown, e.g. when feature list only consists of trendscore
        pass

    if ('trendscore' in feature_list):
        X = np.append(X, data['trendscore'].values.reshape(-1, 1), axis=1)

    y = scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)
    y_dir = data['next_direction'].values.reshape(-1, 1)
    y_dir = np.append(data['stock'].values.reshape(-1, 1), y_dir, axis=1)
    X_train, X_val, X_test = group_by_stock(X)
    y_train, y_val, y_test = group_by_stock(y)
    y_train_dir, y_val_dir, y_test_dir = group_by_stock(y_dir)
    return (X_train, X_val, X_test), (y_train, y_val, y_test), (y_train_dir, y_val_dir, y_test_dir) , scaler_y
