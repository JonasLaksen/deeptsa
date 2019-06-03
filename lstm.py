import csv
import os
import random
import sys
from itertools import combinations

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from models import bidir_lstm_seq
from models.spec_network import SpecializedNetwork
from models.stacked_lstm import StackedLSTM
from models.stacked_lstm_modified import StackedLSTM_Modified
from utils import get_feature_list_lags, evaluate, load_data

seed = int(sys.argv[1]) if sys.argv[1] else 0
type_search = sys.argv[2] if sys.argv[2] else 'hyper'
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


# results = pandas.DataFrame.from_csv('loss-history20k.csv', header=None)
# plot_one('', [results.iloc[[0]].values[0, 100:], results.iloc[[1]].values[0, 100:]],
#          ['Training loss', 'Validation loss'], ['Epoch', 'MAE loss'])


def main(gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False, model_generator=StackedLSTM, layer_sizes=[41],
         copy_weights_from_gen_to_spec=False, feature_list=[], optimizer=Adam(.01), dropout=.2, filename='test',
         loss='MAE', **_):
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    (y_train_dir, y_val_dir, y_test_dir), \
    scaler_y = load_data(feature_list)

    n_features = X_train.shape[2]
    batch_size = X_train.shape[0]
    is_bidir = model_generator is not StackedLSTM
    zero_states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2 * (2 if is_bidir else 1)
    stock_list = [np.arange(len(X_train)).reshape((len(X_train), 1, 1))]

    gen_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False, dropout=dropout)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    # Create the general model

    gen_model.compile(optimizer=optimizer, loss=loss)
    history = gen_model.fit([X_train] + zero_states, y_train, validation_data=([X_val] + zero_states, y_val),
                            epochs=gen_epochs,
                            verbose=1,
                            shuffle=False,
                            batch_size=batch_size,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=500),
                                       ModelCheckpoint(filepath="best-weights.hdf5", verbose=0, save_weights_only=True,
                                                       save_best_only=True)])
    gen_model.load_weights("best-weights.hdf5")

    # plot('test', ['ok'], [history.history['loss']], [history.history['val_loss']])

    # write_to_csv(f'loss-history.csv', history.history)

    gen_pred_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=True,
                                     dropout=dropout)
    gen_pred_model.set_weights(gen_model.get_weights())

    # Create the context model, set the decoder = the gen model
    decoder = model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=True, dropout=dropout)
    if copy_weights_from_gen_to_spec:
        decoder.set_weights(gen_model.get_weights())
    spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(X_train), layer_sizes=layer_sizes,
                                    decoder=decoder, is_bidir=is_bidir)

    spec_model.compile(optimizer=optimizer, loss=loss)
    if load_spec:
        spec_model.load_weights('weights/spec.h5')
        print('Loaded specialised model')

    spec_model.fit([X_train] + stock_list, y_train, validation_data=([X_val] + stock_list, y_val),
                   batch_size=batch_size, epochs=spec_epochs * 20, shuffle=False,
                   callbacks=[ModelCheckpoint('weights/spec.h5', period=1, save_weights_only=True),
                              EarlyStopping(monitor='val_loss', patience=20)])
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

        with open(f"hyperparameter_search/{type_search}_{seed}", "a") as file:
            writer = csv.writer(file)
            if type_search == 'hyper':
                writer.writerow(list(evaluation.values()) + [dropout, layer_sizes, loss])
            if type_search == 'feature':
                writer.writerow(list(evaluation.values()) + feature_list)

        # plot('Train', np.array(stock_list).reshape(-1)[0:3], result_train[0:3], y_train_inv[0:3])
        # plot('Val', np.array(stock_list).reshape(-1)[0:3], result_val[0:3], y_val_inv[0:3])

        # training = {f'training {"spec" if has_context else "gen"}': result_train.tolist(), 'y': y_train_inv.tolist()}
        # validation = {f'validation {"spec" if has_context else "gen"}': result_val.tolist(), 'y': y_val_inv.tolist()}
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/training/{filename}', training)
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/validation/{filename}', validation)


# feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
trading_features = [['price', 'volume'], ['open', 'high', 'low'], ['direction']]
sentiment_features = [['positive_prop', 'negative_prop', 'neutral_prop']]
trendscore_features = [['trendscore']]
s = trading_features + sentiment_features + trendscore_features
temp = sum(map(lambda r: list(combinations(s, r)), range(1, len(s) + 1)), [])
feature_subsets = list(map(lambda x: sum(x, []), temp))


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


def feature_search(other_args):
    features_list = {'feature_list': [['price'],
                                      ['price', 'open', 'high', 'low', 'direction'],
                                      ['price', 'positive_prop', 'negative_prop', 'neutral_prop'],
                                      ['price', 'trendscore'],
                                      ['price', 'open', 'high', 'low', 'direction', 'positive_prop', 'negative_prop',
                                       'neutral_prop', 'trendscore']]}
    arguments_list = [{**other_args, **{i: j}} for i in features_list.keys() for j in features_list[i]]
    for args in arguments_list:
        print({k: args[k] for k in features_list.keys() if k in args})
        main(**args,
             model_generator=StackedLSTM if other_args['model'] == 'stacked' else bidir_lstm_seq.build_model,
             filename='test')


def main2(gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False, model_generator=StackedLSTM, layer_sizes=[41],
          copy_weights_from_gen_to_spec=False, feature_list=[], optimizer=Adam(.01), dropout=.2, filename='test',
          loss='MAE', **_):
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    (y_train_dir, y_val_dir, y_test_dir), \
    scaler_y = load_data(feature_list)

    n_features = X_train.shape[2]
    batch_size = X_train.shape[0]
    is_bidir = model_generator is (not StackedLSTM or not StackedLSTM_Modified)
    zero_states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2 * (2 if is_bidir else 1)
    stock_list = [np.arange(len(X_train)).reshape((len(X_train), 1, 1))]

    gen_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                return_states=False, dropout=dropout)

    gen_model.compile(optimizer=optimizer, loss=loss)
    history = gen_model.fit([X_train] + zero_states, [y_train, y_train_dir], validation_data=([X_val] + zero_states,
                                                                                              [y_val, y_val_dir]),
                            epochs=gen_epochs * 200,
                            verbose=1,
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

    for model in ([gen_pred_model] if gen_epochs > 0 else []):
        has_context = isinstance(model, SpecializedNetwork)
        # If general model, give zeros as input, if context give stock ids as input
        init_state = model.encoder.predict(stock_list) if has_context else zero_states

        if has_context:
            model = model.decoder

        result_train, result_train_dir, *new_states = model.predict([X_train] + init_state)
        result_val, result_val_dir, *new_states = model.predict([X_val] + new_states)
        result_test, result_test_dir, *_ = model.predict([X_test] + new_states)

        result_train, result_val, result_test, y_train_inv, y_val_inv, y_test_inv = map(
            # lambda x: scaler_y.inverse_transform(x).reshape(-1),
            lambda x: np.array(list(map(scaler_y.inverse_transform, x))),
            [result_train, result_val, result_test, y_train, y_val, y_test])

        evaluation = evaluate(result_val, y_val_inv)
        with open(f"hyperparameter_search/{seed}", "a") as file:
            writer = csv.writer(file)
            writer.writerow(list(evaluation.values()) + [dropout, layer_sizes, loss])
            # writer.writerow(list(evaluation.values()) + feature_list)

        # plot('Train', np.array(stock_list).reshape(-1)[0:3], result_train[0:3], y_train_inv[0:3])
        # plot('Val', np.array(stock_list).reshape(-1)[0:3], result_val[0:3], y_val_inv[0:3])

        # training = {f'training {"spec" if has_context else "gen"}': result_train.tolist(), 'y': y_train_inv.tolist()}
        # validation = {f'validation {"spec" if has_context else "gen"}': result_val.tolist(), 'y': y_val_inv.tolist()}
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/training/{filename}', training)
        # write_to_csv(f'plot_data/{"spec" if has_context else "gen"}/validation/{filename}', validation)


# main2(**arguments,
#                      model_generator=StackedLSTM_Modified,
#                      filename='test2')

arguments = {
    'copy_weights_from_gen_to_spec': False,
    'feature_list': sum(trading_features + sentiment_features + trendscore_features, []),
    'gen_epochs': 5000,
    'spec_epochs': 0,
    'load_gen': False,
    'load_spec': False,
    'dropout': .2,
    'layer_sizes': [128],
    'optimizer': Adam(.001),
    'loss': 'MAE',
    'model': 'stacked',
    # 'model': 'bidir'
}
if type_search == 'hyper':
    # Hyperparameter search
    print('hyper search')
    possible_hyperparameters = {
        'dropout': [0, .2, .5],
        'layer_sizes': [[32], [128], [160]],
        'loss': ['MAE', 'MSE']
    }
    hyperparameter_search(possible_hyperparameters, arguments)
elif type_search == 'feature':
    # Feature search
    print('feature search')
    feature_search(arguments)
