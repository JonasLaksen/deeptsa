from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import MAE, MAPE
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn import preprocessing

import models.bidir_lstm_seq as bidir_lstm_seq
from models.spec_network import SpecializedNetwork
from models.stacked_lstm import StackedLSTM
from utils import get_feature_list_lags, group_by_stock, expand, plot


def main(bidir_seq_epoch=0, gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False):
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()

    # feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
    feature_list = ['volume', 'price']
    feature_list = get_feature_list_lags(feature_list, lags=2)
    feature_list = feature_list + ['open', 'high', 'low']

    n_features = len(feature_list)
    layer_sizes = [41] * 3
    batch_size = 17

    scaler_X = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.MinMaxScaler()
    X = scaler_X.fit_transform(data[feature_list].values)
    y = scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)

    X_no_pad, X, X_test = group_by_stock(X, n_features)
    y_no_pad, y, y_test = group_by_stock(y, 1)
    stock_id = 10

    X_train, X_test = expand(X_no_pad[stock_id]), expand(X_test[stock_id])
    y_train = scaler_y.inverse_transform(y_no_pad[stock_id]).reshape(-1)
    y_test = scaler_y.inverse_transform(y_test[stock_id]).reshape(-1)

    gen_model = StackedLSTM(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size, return_states=False)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    if gen_epochs > 0:
        states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2
        gen_model.compile(optimizer=Adam(0.001), loss=MAE)
        gen_model.fit([X] + states, y, batch_size=batch_size, epochs=gen_epochs, shuffle=False,
                      callbacks=[
                          # Save model every 10th epoch
                          ModelCheckpoint('weights/gen.h5', period=10, save_weights_only=True),
                          # Write to logfile to see graph in TensorBoard
                          # TensorBoard(log_dir='logs/gen/{}'.format(datetime.now()))
                      ])

        gen_pred_model = StackedLSTM(n_features=n_features, layer_sizes=layer_sizes, batch_size=1,
                                     return_states=True)
        gen_pred_model.set_weights(gen_model.get_weights())

        states = [np.zeros((1, layer_sizes[0]))] * len(layer_sizes) * 2
        result_train, *new_states = gen_pred_model.predict([X_train] + states)
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test, *_ = gen_pred_model.predict([X_test] + new_states)
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]

    if spec_epochs > 0:
        spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                        batch_size=batch_size, init_all_layers=True)
        spec_model.decoder.set_weights(gen_model.get_weights())
        spec_model.compile(optimizer=Adam(0.001), loss=MAE)
        if load_spec:
            spec_model.load_weights('weights/spec.h5')
            print('Loaded specialised model')

        [plot_model(x[1], 'model_plots/{}.png'.format(x[0]), show_shapes=True) for x in
         [('gen', gen_model), ('spec', spec_model), ('encoder', spec_model.encoder)]]

        X_stock_list = np.arange(len(X)).reshape((len(X), 1, 1))
        spec_model.fit([X, X_stock_list], y, batch_size=batch_size, epochs=spec_epochs, shuffle=False,
                       callbacks=[
                           ModelCheckpoint('weights/spec.h5', period=1, save_weights_only=True),
                           TensorBoard(log_dir='logs/spec/{}'.format(datetime.now()))])

        spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                             batch_size=1, init_all_layers=True, return_states=True)
        spec_pred_model.set_weights(spec_model.get_weights())

        result_train, *new_states = spec_pred_model.predict([X_train, np.array([[[stock_id]]])])
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test, *_ = spec_pred_model.decoder.predict([X_test] + new_states)
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        print(result_train[-5:])
        print(result_test[:5])

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]

    if bidir_seq_epoch > 0:
        layer_size = 97
        states = [np.zeros((batch_size, layer_size))] * 4
        bidir_model = bidir_lstm_seq.build_model(n_features=n_features, layer_size=layer_size, stateful=False,
                                                 batch_size=batch_size, return_states=False)

        bidir_model.compile(optimizer=Adam(0.001), loss=MAE, metrics=[MAPE])

        bidir_model.fit([X] + states, y, batch_size=batch_size, epochs=bidir_seq_epoch, shuffle=False,
                        callbacks=[
                            # Save model every 10th epoch
                            # ModelCheckpoint('weights/bidir.h5', period=10, save_weights_only=True),
                            # Write to logfile to see graph in TensorBoard
                            # TensorBoard(log_dir='logs/gen/{}'.format(datetime.now()))
                        ])

        bidir_pred_model = bidir_lstm_seq.build_model(n_features=n_features, layer_size=layer_size, stateful=True,
                                                      batch_size=1, return_states=True)
        bidir_pred_model.set_weights(bidir_model.get_weights())

        result_train, *new_states = bidir_pred_model.predict([X_train] + states)
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test, *_ = bidir_pred_model.predict([X_test] + new_states)
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]

    plt.show()


main(bidir_seq_epoch=250, gen_epochs=0, spec_epochs=0, load_gen=False, load_spec=False)
