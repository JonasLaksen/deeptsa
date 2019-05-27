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


def expand(x): return np.expand_dims(x, axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics(result, y):
    result = np.asarray(result).reshape((result.shape[0], -1))
    y = np.asarray(y).reshape((result.shape[0], -1))
    mape = mean_absolute_percentage_error(y, result)
    mae = mean_absolute_error(y, result)
    mse = mean_squared_error(y, result)
    accuracy_direction = mean_direction_eval(result, y)
    print({'MAPE': mape, 'MAE': mae, 'MSE': mse, 'DA': accuracy_direction})

def plot(title, result, y):
    # result = np.asarray(result)
    # y = np.asarray(y)
    # mape = mean_absolute_percentage_error(y, result)
    # mae = mean_absolute_error(y, result)
    # mse = mean_squared_error(y, result)
    # accuracy_direction = direction_eval(result, y)
    # print(mape, mae, mse, accuracy_direction)
    # pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b', title=title)
    # pd.DataFrame({'Actual': y})['Actual'].plot(label='Actual', c='r', linestyle='--')
    pyplot.plot(result)
    pyplot.plot(y)
    pyplot.show()


def direction_value(x, y):
    if x > y:
        return -1
    elif x < y:
        return 1
    else:
        return 0

def mean_direction_eval(result, y):
    return np.mean(np.array(list(map(lambda x: direction_eval(x[0], x[1]), zip(result, y)))))

def direction_eval(result, y):
    result_pair = list(map(lambda x, y: direction_value(x, y), result[:-1], result[1:]))
    y_pair = list(map(lambda x, y: direction_value(x, y), y[:-1], y[1:]))
    return accuracy_score(y_pair, result_pair)


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
