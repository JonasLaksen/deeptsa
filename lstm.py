import copy
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import MAE
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn import preprocessing

from models.gen_network import GeneralizedNetwork
from models.spec_network import SpecializedNetwork


def expand(x): return np.expand_dims(x, axis=0)


def plot(result, y):
    pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b')
    pd.DataFrame({'Actual': y})['Actual'].plot(label='Actual', c='r', linestyle='--')


def group_by_stock(data, n_features):
    group_by_dict = {}
    for row in data:
        try:
            group_by_dict[row[0]].append(row[1:])
        except:
            group_by_dict[row[0]] = [row[1:]]

    data = list(map(lambda x: np.array(group_by_dict[x])[:-50], group_by_dict.keys()))
    test_data = list(map(lambda x: np.array(group_by_dict[x])[-50:], group_by_dict.keys()))

    data_padded = copy.deepcopy(data)

    max_length = np.max([x.shape[0] for x in data])
    for i in range(len(data)):
        data_padded[i] = np.append(data_padded[i], np.zeros((max_length - data_padded[i].shape[0], n_features)), axis=0)

    return np.array(data), np.array(data_padded), np.array(test_data)


def main(mode='gen', both=False, gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False):
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()

    # feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
    feature_list = ['volume', 'price']

    n_features = len(feature_list)
    layer_sizes = [41, 41, 41]
    batch_size = 17

    scaler_X = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.MinMaxScaler()
    X = scaler_X.fit_transform(data[feature_list].values)
    y = scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)

    X_no_pad, X, X_test = group_by_stock(X, n_features)
    y_no_pad, y, y_test = group_by_stock(y, 1)
    stock_id = 12

    X_train, X_test = expand(X_no_pad[stock_id]), expand(X_test[stock_id])
    y_train = scaler_y.inverse_transform(y_no_pad[stock_id]).reshape(-1)
    y_test = scaler_y.inverse_transform(y_test[stock_id]).reshape(-1)

    gen_model = GeneralizedNetwork(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                   stateful=False)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    if mode == 'gen' or both:
        gen_model.compile(optimizer=Adam(0.001), loss=MAE)
        gen_model.fit(X, y, batch_size=batch_size, epochs=gen_epochs, shuffle=False,
                      callbacks=[
                          # Save model every 10th epoch
                          ModelCheckpoint('weights/gen.h5', period=10, save_weights_only=True),
                          # Write to logfile to see graph in TensorBoard
                          TensorBoard(log_dir='logs/gen/{}'.format(datetime.now()))])

        gen_pred_model = GeneralizedNetwork(n_features=n_features, layer_sizes=layer_sizes, batch_size=1, stateful=True)
        gen_pred_model.set_weights(gen_model.get_weights())

        result_train = gen_pred_model.predict(X_train)
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test = gen_pred_model.predict(X_test)
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]

    if mode == 'spec' or both:
        spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                        stateful=False, batch_size=batch_size)
        spec_model.decoder.get_layer('LSTM').set_weights(gen_model.get_layer('LSTM')
                                                                .get_weights())
        if load_spec:
            spec_model.load_weights('weights/spec.h5')
            print('Loaded specialised model')

        [plot_model(x[1], 'model_plots/{}.png'.format(x[0]), show_shapes=True) for x in
         [('gen', gen_model), ('spec', spec_model), ('decoder', spec_model.decoder), ('encoder', spec_model.encoder),
          ('lstm', spec_model.decoder.get_layer('LSTM'))]]

        spec_model.compile(optimizer=Adam(0.001), loss=MAE)

        X_stock_list = np.arange(len(X)).reshape((len(X), 1, 1))
        spec_model.fit([X, X_stock_list], y, batch_size=batch_size, epochs=spec_epochs, shuffle=False,
                       callbacks=[
                           # Save model every 10th epoch
                           ModelCheckpoint('weights/spec.h5', period=10, save_weights_only=True),
                           # Write to logfile to see graph in TensorBoard
                           TensorBoard(log_dir='logs/spec/{}'.format(datetime.now()))])

        spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                             stateful=True, batch_size=1)
        spec_pred_model.set_weights(spec_model.get_weights())

        result_train = spec_pred_model.predict([X_train, np.array([[[stock_id]]])])
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test = spec_pred_model.predict([X_test, np.array([[[stock_id]]])])
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]
    plt.show()


main('spec', both=False, gen_epochs=100, spec_epochs=11, load_gen=False, load_spec=False)
