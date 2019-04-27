import copy
import keras
import keras.layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense, Reshape, Embedding, Masking
from sklearn import preprocessing


def expand(x): np.expand_dims(x, axis=0)


def plot(result, y):
    pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b')
    pd.DataFrame({'Actual': y})['Actual'].plot(label='Actual', c='r', linestyle='--')


def build_gen_network(n_features, layer_sizes, stateful, batch_size):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
    for i, layer_size in enumerate(layer_sizes):
        model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful, dropout=.2))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MAE,
        metrics=[keras.metrics.MAPE]
    )

    return model


def build_encoder(num_stocks, layer_sizes):
    encoder_inputs = keras.layers.Input(shape=(1, 1))

    encoder_reshape = Reshape((layer_sizes[0],))
    init_h = Embedding(num_stocks + 1, layer_sizes[0])(encoder_inputs)
    init_h = encoder_reshape(init_h)

    init_c = Embedding(num_stocks + 1, layer_sizes[0])(encoder_inputs)
    init_c = encoder_reshape(init_c)

    return keras.Model(encoder_inputs, [init_h, init_c])


def build_decoder(n_features, layer_sizes, stateful, batch_size):
    X = Input(batch_shape=(batch_size, None, n_features))
    h_input, c_input = Input(shape=(layer_sizes[0],)), Input(shape=(layer_sizes[0],))

    masked = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features))
    state_h, state_c = h_input, c_input
    output = masked(X)

    for size in layer_sizes:
        lstm = LSTM(size, return_sequences=True, return_state=True, stateful=stateful, dropout=.2)
        output, state_h, state_c = lstm(output, initial_state=[state_h, state_c])

    next_price = Dense(1, activation='linear')(output)

    model = keras.Model([X, h_input, c_input], [next_price, state_h, state_c])
    return model


def build_spec_network(n_features, num_stocks, layer_sizes, stateful, batch_size, gen_model, spec_model):
    stock_id = Input(shape=(1, 1))
    encoder = build_encoder(num_stocks, layer_sizes)
    state_h, state_c = encoder(stock_id)

    X = Input(batch_shape=(batch_size, None, n_features))
    decoder = build_decoder(n_features, layer_sizes, stateful, batch_size)
    next_price, state_h, state_c = decoder([X, state_h, state_c])

    model = keras.Model([X, stock_id], next_price)

    copy_weights_layers = [4, 5, 6, 7]
    for i, layer_i in enumerate(copy_weights_layers):
        decoder.layers[layer_i].set_weights(gen_model.layers[i + 1].get_weights())

    if spec_model:
        model.set_weights(spec_model.get_weights())

    return model, encoder, decoder


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


def main(mode='gen', both=False, gen_epochs=0, spec_epochs=0):
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()

    # feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
    feature_list = ['open', 'high', 'low', 'volume', 'price']

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

    try:
        gen_model = load_model('model/gen.h5')
        print('Loaded generalised model')
    except OSError:
        print('Couldn\'t load generalised model')
        gen_model = None

    if mode == 'gen' or both:
        if not gen_model:
            print('Building new generalised model')
            gen_model = build_gen_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                          stateful=False)

        # Save model every epoch
        gen_model.fit(X, y, batch_size=batch_size, epochs=gen_epochs, shuffle=False,
                      callbacks=[ModelCheckpoint('model/gen.h5')])

        gen_pred_model = build_gen_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=1, stateful=True)
        gen_pred_model.set_weights(gen_model.get_weights())

        result_train = gen_pred_model.predict(X_train)
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test = gen_pred_model.predict(X_test)
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]

    if mode == 'spec' or both:
        spec_model, encoder, decoder = build_spec_network(n_features=n_features, num_stocks=len(y),
                                                          layer_sizes=layer_sizes, stateful=False, gen_model=gen_model,
                                                          spec_model=None, batch_size=batch_size)

        spec_model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss=keras.losses.MAE,
            metrics=[keras.metrics.MAPE]
        )

        X_stock_list = np.arange(len(X)).reshape((len(X), 1, 1))
        spec_model.fit([X, X_stock_list], y, batch_size=batch_size, epochs=spec_epochs, shuffle=False)

        spec_pred_model, _, _ = build_spec_network(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                                   stateful=True, gen_model=gen_model, spec_model=spec_model,
                                                   batch_size=1)

        result_train = spec_pred_model.predict([X_train, np.array([[[stock_id]]])])
        result_train = scaler_y.inverse_transform(result_train[0]).reshape(-1)

        result_test = spec_pred_model.predict([X_test, np.array([[[stock_id]]])])
        result_test = scaler_y.inverse_transform(result_test[0]).reshape(-1)

        [plot(*x) for x in [(result_train, y_train), (result_test, y_test)]]
    plt.show()


main('spec', False, gen_epochs=0, spec_epochs=10)