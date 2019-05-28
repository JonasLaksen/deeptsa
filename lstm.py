import csv
import os
import random
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from models import bidir_lstm_seq
from models.spec_network import SpecializedNetwork
from models.stacked_lstm import StackedLSTM
from utils import get_feature_list_lags, group_by_stock, evaluate

seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


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

    X_train, X_val, X_test = group_by_stock(X)
    y_train, y_val, y_test = group_by_stock(y)
    return (X_train, X_val, X_test), (y_train, y_val, y_test), scaler_y


def main(gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False, model_generator=StackedLSTM, layer_sizes=[41],
         copy_weights_from_gen_to_spec=False, feature_list=[], optimizer=Adam(.01), dropout=.2, filename='test',
         loss='MAE', **_):
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    scaler_y = load_data(feature_list)

    n_features = X_train.shape[2]
    batch_size = X_train.shape[0]
    is_bidir = model_generator is not StackedLSTM
    zero_states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2 * (2 if is_bidir else 1)
    stock_list = [np.arange(len(X_train)).reshape((len(X_train), 1, 1))]

    gen_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                return_states=False, dropout=dropout)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    # Create the general model
    gen_model.compile(optimizer=optimizer, loss=loss)
    history = gen_model.fit([X_train] + zero_states, y_train, validation_data=([X_val] + zero_states, y_val),
                            epochs=gen_epochs * 1000,
                            verbose=0,
                            shuffle=False,
                            batch_size=batch_size,
                            callbacks=[ModelCheckpoint('weights/gen.h5', period=10, save_weights_only=True),
                                       EarlyStopping(monitor='val_loss', patience=20)])

    # write_to_csv(f'plot_data/gen/loss/{filename}.csv', history.history)

    gen_pred_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                     return_states=True, dropout=dropout)
    gen_pred_model.set_weights(gen_model.get_weights())

    # Create the context model, set the decoder = the gen model
    decoder = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size, return_states=True,
                              dropout=dropout)
    if copy_weights_from_gen_to_spec:
        decoder.set_weights(gen_model.get_weights())
    spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(X_train), layer_sizes=layer_sizes,
                                    decoder=decoder, is_bidir=is_bidir)
    spec_model.compile(optimizer=optimizer, loss=loss)
    if load_spec:
        spec_model.load_weights('weights/spec.h5')
        print('Loaded specialised model')

    spec_model.fit([X_train] + stock_list, y_train, validation_data=([X_val] + stock_list, y_val),
                   batch_size=batch_size, epochs=spec_epochs, shuffle=False,
                   callbacks=[ModelCheckpoint('weights/spec.h5', period=1, save_weights_only=True)])
    # write_to_csv(f'plot_data/spec/loss/{filename}.csv', history.history)
    spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(X_train), layer_sizes=layer_sizes,
                                         return_states=True, decoder=spec_model.decoder, is_bidir=is_bidir)
    spec_pred_model.set_weights(spec_model.get_weights())

    # The place for saving stuff for plotting
    # Only plot if the epoch > 0
    for model in ([gen_pred_model] if gen_epochs > 0 else []) + ([spec_pred_model] if spec_epochs > 0 else []):
        has_context = isinstance(model, SpecializedNetwork)
        # If general model, give zeros as input, if context give stock ids as input
        init_state = model.encoder.predict(stock_list) if has_context else zero_states

        if has_context:
            model = model.decoder

        result_train, *new_states = model.predict([X_train] + init_state)
        result_val, *new_states = model.predict([X_val] + new_states)
        result_test, *_ = model.predict([X_test] + new_states)

        result_train, result_val, result_test, y_train_inv, y_val_inv, y_test_inv = map(
            # lambda x: scaler_y.inverse_transform(x).reshape(-1),
            lambda x: np.array(list(map(scaler_y.inverse_transform, x))),
            [result_train, result_val, result_test, y_train, y_val, y_test])

        evaluation = evaluate(result_val, y_val_inv)
        # with open(f"hyperparameter_search/{seed}", "a") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(list(evaluation.values()) + [dropout, layer_sizes, loss])

        with open(f"hyperparameter_search/features_{seed}", "a") as file:
            writer = csv.writer(file)
            writer.writerow(list(evaluation.values()) + feature_list)

        # plot('Train', np.array(stock_list).reshape(-1)[0:3], result_train[0:3], y_train_inv[0:3])
        # plot('Val', np.array(stock_list).reshape(-1)[0:3], result_val[0:3], y_val_inv[0:3])
        # training = {f'training {"spec" if has_context else "gen"}': result_train.tolist(), 'y': y_train_inv.tolist()}
        # validation = {f'validation {"spec" if has_context else "gen"}': result_val.tolist(), 'y': y_val_inv.tolist()}
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/training/{filename}', training)
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/validation/{filename}', validation)


# feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
feature_list = ['volume', 'price']
feature_list = get_feature_list_lags(feature_list, lags=2)
feature_list = feature_list + ['positive', 'negative', 'neutral', 'positive_prop', 'negative_prop', 'neutral_prop',
                               'trendscore', 'open', 'high', 'low']
trading_features = [['price', 'volume'], ['open', 'high', 'low']]
sentiment_features = [['positive', 'negative', 'neutral'], ['positive_prop', 'negative_prop', 'neutral_prop']]
trendscore_features = [['trendscore']]
s = trading_features + sentiment_features + trendscore_features
temp = sum(map(lambda r: list(combinations(s, r)), range(1, len(s) + 1)), [])
feature_subsets = list(map(lambda x: sum(x, []), temp))

arguments = {
    'copy_weights_from_gen_to_spec': False,
    'feature_list': sum(trading_features + sentiment_features + trendscore_features, []),
    'gen_epochs': 1,
    'spec_epochs': 0,
    'load_gen': False,
    'load_spec': False,
    'model': 'stacked',
    'dropout': .2,
    'layer_sizes': [128],
    'optimizer': Adam(.001),
    'loss': 'MAE'
    # 'model': 'bidir',
}

# Hyperparameter search
# possible_hyperparameters = {
#     'dropout': [0, .2, .5],
#     'layer_sizes': [[32], [128], [160]],
#     'loss': ['MAE', 'MSE']
# }

# Feature search
possible_hyperparameters = {
    'feature_list': feature_subsets
}


try:
    os.remove(f'{seed}')
except OSError:
    pass

def hyperparameter_search(possible, other_args):
    for i in possible['dropout']:
        for j in possible['layer_sizes']:
            for k in possible['loss']:
                args = other_args
                args['dropout'] = i
                args['layer_sizes'] = j
                args['loss'] = k
                print({k: args[k] for k in possible_hyperparameters.keys() if k in args})
                main(**args,
                     model_generator=StackedLSTM if other_args['model'] == 'stacked' else bidir_lstm_seq.build_model,
                     filename='test')

def feature_search(possible, other_args):
    arguments_list = [{**other_args, **{i: j}} for i in possible.keys() for j in possible[i] ]
    for args in arguments_list:
        print({k: args[k] for k in possible.keys() if k in args})
        main(**args,
             model_generator=StackedLSTM if other_args['model'] == 'stacked' else bidir_lstm_seq.build_model,
             filename='test')


feature_search(possible_hyperparameters, arguments)
