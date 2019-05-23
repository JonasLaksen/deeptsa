from copy import deepcopy
from itertools import combinations

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import MAE
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from models import bidir_lstm_seq
from models.spec_network import SpecializedNetwork
from models.stacked_lstm import StackedLSTM
from utils import get_feature_list_lags, group_by_stock, print_metrics


def load_data(feature_list):
    data = pd.read_csv('dataset_v2.csv', index_col=0)
    data = data.dropna()
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(data[feature_list].values)
    y = scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)

    X_train, X_val, X_test = group_by_stock(X)
    y_train, y_val, y_test = group_by_stock(y)
    return (X_train, X_val, X_test), (y_train, y_val, y_test), scaler_y


def main(gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False, model_generator=StackedLSTM, layer_sizes=[41],
         copy_weights_from_gen_to_spec=False, feature_list=[], learning_rate=.001, dropout=.2, filename='test', **_):
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    scaler_y = load_data(feature_list)

    n_features = X_train.shape[2]
    batch_size = X_train.shape[0]
    is_bidir = model_generator is not StackedLSTM
    zero_states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2 * (2 if is_bidir else 1)
    stock_list = [np.arange(len(X_train)).reshape((len(X_train), 1, 1))]

    gen_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                return_states=False)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    # Create the general model
    gen_model.compile(optimizer=Adam(learning_rate), loss=MAE)
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
    spec_model.compile(optimizer=Adam(learning_rate), loss=MAE)
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

        print_metrics(result_val, y_val_inv)
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
    'feature_list': feature_list,
    'gen_epochs': 1,
    'spec_epochs': 0,
    'load_gen': False,
    'load_spec': False,
    'model': 'stacked',
    'dropout': .2,
    'layer_sizes': [64],
    'learning_rate': .01
    # 'model': 'bidir',
}

# Hyperparameter search
# possible_hyperparameters = {
#     'dropout': [0, .2, .5],
#     'layer_sizes': [[8], [64], [256]],
#     'learning_rate': [.001, .01, .1],
# }

# Feature search
possible_hyperparameters = {
    'feature_list': feature_subsets
}


def hyperparameter_search(possible, other_args):
    for key in possible_hyperparameters.keys():
        for possible_value in possible[key]:
            args = deepcopy(other_args)
            args[key] = possible_value
            print({k: args[k] for k in possible_hyperparameters.keys() if k in args})
            main(**args,
                 model_generator=StackedLSTM if other_args['model'] == 'stacked' else bidir_lstm_seq.build_model,
                 filename='test')


hyperparameter_search(possible_hyperparameters, arguments)
