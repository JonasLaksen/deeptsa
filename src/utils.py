import copy
import csv
import json
from base64 import b64encode, b64decode
from itertools import combinations
from zlib import compress, decompress

import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from src.constants import stock_list as s_l


def expand(x): return np.expand_dims(x, axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + .0001))) * 100


def evaluate(result, y, y_type = 'next_change'):
    # result = np.asarray(result).reshape((result.shape[0], -1))
    # y = np.asarray(y).reshape((result.shape[0], -1))
    mape = mean_absolute_percentage_error(y, result)
    mae = mean_absolute_error(y, result)
    mse = mean_squared_error(y, result)
    accuracy_direction = mean_direction_eval(result, y, y_type)
    return {'MAPE': mape, 'MAE': mae, 'MSE': mse, 'DA': accuracy_direction}


def plot(directory, title, stocklist, result, y, legends=['Predicted', 'True value'], axises=['Day', 'Price $'], ):
    [plot_one(f'{title}: {stocklist[i]}', [result[i], y[i]], legends, axises, f'{directory}/{title}-{i}.png') for i in
     range(len(result))]


def plot_one(title, xs, legends, axises, filename = ''):
    assert len(xs) == len(legends)
    pyplot.title(title)
    [pyplot.plot(x, label=legends[i]) for i, x in enumerate(xs)]
    pyplot.legend(loc='upper left')
    pyplot.xlabel(axises[0])
    pyplot.ylabel(axises[1])

    if(len(filename) > 0):
        pyplot.savefig(filename, bbox_inches='tight')
    pyplot.show()
    pyplot.close()


def direction_value(x, y):
    if x > y:
        return [0]
    else:
        return [1]


def mean_direction_eval(result, y, y_type):
    return np.mean(np.array(list(map(lambda x: direction_eval(x[0], x[1], y_type), zip(result, y)))))


def direction_eval(result, y, y_type):
    if y_type == 'next_change':
        n_same_dir = sum(list(map(lambda x,y: 1 if (x >= 0 and y >= 0) or (x < 0 and y<0) else 0, result, y)))
        return n_same_dir/len(result)

    result_pair = list(map(lambda x, y: direction_value(x, y), y[:-1], result[1:]))
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


def group_by_stock(data):
    group_by_dict = {}
    for row in data:
        try:
            group_by_dict[row[0]].append(row[0:])
        except:
            group_by_dict[row[0]] = [row[0:]]

    group_by_dict = {k: v for k, v in group_by_dict.items() if len(v) > 1600}
    data_size = len(min(group_by_dict.values(), key=len))
    # data_size = 1661
    data = list(map(lambda x: np.array(group_by_dict[x][-data_size:]), group_by_dict.keys()))
    return np.array(data)


def get_feature_list_lags(features, lags=0):
    all_features = copy.deepcopy(features)

    for feature in features:
        for i in range(lags):
            all_features = all_features + ['prev_' + feature + '_' + str(i)]

    return all_features


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


def load_data(feature_list, y_type, train_portion, remove_portion_at_end, should_scale_y=True):
    data = pd.read_csv('dataset_v2.csv', index_col=0)
    data = data.dropna()
    # data = data[data['stock'] == 'AAPL']
    data['all_positive'] = data.groupby('date')['positive'].sum()
    data['all_negative'] = data.groupby('date')['negative'].sum()
    data['all_neutral'] = data.groupby('date')['neutral'].sum()
    feature_list_element_not_in_dataset = set(feature_list) - set(data.columns.values)
    if(len(feature_list_element_not_in_dataset) > 0):
        raise Exception(f'En feature ligger ikke i datasettet {feature_list_element_not_in_dataset}')
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() \
        if should_scale_y \
        else FunctionTransformer(lambda x:x, lambda x:x)

    X = data['stock'].values.reshape(-1, 1)

    try:
        values = data[[x for x in feature_list if x is not 'trendscore']].values
        X = np.append(X, values, axis=1)
    except:
        # If there are no features to be scaled an error is thrown, e.g. when feature list only consists of trendscore
        pass

    if ('trendscore' in feature_list):
        X = np.append(X, data['trendscore'].values.reshape(-1, 1), axis=1)

    y = data[y_type].values.reshape(-1, 1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)
    y = group_by_stock(y)
    y_dir = data['next_direction'].values.reshape(-1, 1)
    y_dir = np.append(data['stock'].values.reshape(-1, 1), y_dir, axis=1)
    y_dir = group_by_stock(y_dir)

    X = group_by_stock(X)
    train_size = int(X.shape[1]*train_portion)
    remove_size = int(X.shape[1]*remove_portion_at_end)

    X_train = X[:,:train_size,1:]
    X_test = X[:,train_size:-remove_size,1:]
    y_train = y[:,:train_size, 1:]
    y_test = y[:,train_size:-remove_size, 1:]
    for i in range(X_train.shape[0]):
        scaler_X.fit(X_train[i])
        if should_scale_y:
            scaler_y.fit(y_train[i])
    for i in range(X_train.shape[0]):
        X_train[i] = scaler_X.transform(X_train[i])
        if should_scale_y:
            y_train[i] = scaler_y.transform(y_train[i])
    for i in range(X_train.shape[0]):
        X_test[i] = scaler_X.transform(X_test[i])
        if should_scale_y:
            y_test[i] = scaler_y.transform(y_test[i])

    if(X_train.shape[2] != len(feature_list)):
        raise Exception('Lengden er feil')

    return X_train.astype(np.float), y_train.astype(np.float), \
           X_test.astype(np.float), y_test.astype(np.float), \
           y_dir[:,:,1:], \
           X[:, 0,0], \
           scaler_y


trading_features = [['price', 'volume', 'change'], ['open', 'high', 'low'], ['direction']]
sentiment_features = [['positive', 'negative', 'neutral'], ['positive_prop', 'negative_prop', 'neutral_prop']]#, ['all_positive', 'all_negative', 'all_neutral']]
trendscore_features = [['trendscore']]
s = trading_features + sentiment_features + trendscore_features
temp = sum(map(lambda r: list(combinations(s, r)), range(1, len(s) + 1)), [])
feature_subsets = list(map(lambda x: sum(x, []), temp))


def get_features(trading=True, sentiment=True, trendscore=True):
    list = []
    if (trading):
        list.extend(trading_features)
    if (sentiment):
        list.extend(sentiment_features)
    if (trendscore):
        list.extend(trendscore_features)
    return sum(list, [])

def flatten_list(l):
    return [item for sublist in l for item in sublist]
