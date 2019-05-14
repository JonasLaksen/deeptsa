import copy

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


def expand(x): return np.expand_dims(x, axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def plot(title, result, y):
    mape = mean_absolute_percentage_error(y, result)
    mae = mean_absolute_error(y, result)
    mse = mean_squared_error(y, result)
    accuracy_direction = direction_eval(result, y)
    print(mape, mae, mse, accuracy_direction)
    pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b', title=title)
    pd.DataFrame({'Actual': y})['Actual'].plot(label='Actual', c='r', linestyle='--')


def direction_value(x, y):
    if x > y:
        return -1
    elif x < y:
        return 1
    else:
        return 0


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

    data_size = len(min(group_by_dict.values(), key=len))
    train_size, val_size, test_size = int(data_size * training_prop), int(data_size * validation_prop), int(
        data_size * test_prop)
    train_data = list(map(lambda x: np.array(group_by_dict[x])[-data_size: -data_size + train_size], group_by_dict.keys()))
    val_data = list(map(lambda x: np.array(group_by_dict[x])[-data_size + train_size: -data_size + train_size + val_size], group_by_dict.keys()))
    test_data = list(map(lambda x: np.array(group_by_dict[x])[-test_size:], group_by_dict.keys()))
    return np.array(train_data), np.array(val_data), np.array(test_data)


def get_feature_list_lags(features, lags=0):
    all_features = copy.deepcopy(features)

    for feature in features:
        for i in range(lags):
            all_features = all_features + ['prev_' + feature + '_' + str(i)]

    return all_features
