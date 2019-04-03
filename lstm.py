import tensorflow as tf
from tensorflow import keras
import pandas as pd
import copy
from sklearn import preprocessing
import matplotlib.pyplot as plt
from constants import stock_list
import numpy as np

def build_network(n_features, layer_sizes, stateful, batch_size, loss='mse'):

    # inputs = keras.layers.Input(batch_shape=(batch_size, None, n_features))
    #
    # input_masking = keras.layers.Masking(mask_value=0.,
    #                                            batch_input_shape=(batch_size, None, n_features))
    #
    # lstms = {}
    # dropouts = {}
    # for i, size in enumerate(layer_sizes):
    #     lstms[i] = keras.layers.LSTM(size, return_sequences=True,
    #                                          return_state=True,
    #                                          stateful=stateful)
    #     dropouts[i] = keras.layers.Dropout(0.2)
    #
    # output_dense = keras.layers.Dense(1, activation='linear')
    #
    # output = input_masking(inputs)
    #
    # for i in range(len(layer_sizes)):
    #     if i == 0:
    #         output, state_h, state_c = lstms[i](output)
    #     else:
    #         output, state_h, state_c = lstms[i](output, initial_state=init_state)
    #     output = dropouts[i](output)
    #     init_state = [state_h, state_c]
    #
    # output = output_dense(output)
    #
    # model = keras.Model(inputs, output)


    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
    for i, layer_size in enumerate(layer_sizes):
        model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MAE,
        metrics=[keras.metrics.MAPE]
    )

    return model

def build_spes_network(n_features, num_stocks, layer_sizes, stateful, batch_size, gen_model, spes_model, loss='mse'):
    encoder_inputs = keras.layers.Input(shape=(1, 1))

    decoder_inputs = keras.layers.Input(batch_shape=(batch_size, None, n_features))

    encoder_embed_h = keras.layers.Embedding(num_stocks + 1, layer_sizes[0])
    encoder_reshape = keras.layers.Reshape((1, layer_sizes[0]), input_shape=(None, 1, 1, layer_sizes[0]))
    # encoder_embed_lambda = keras.layers.Lambda(lambda input: tf.reshape(input, [1,self.gen_dims[1]]))
    encoder_embed_lambda = keras.layers.Lambda(lambda input: tf.reshape(input, [batch_size, layer_sizes[0]]))
    state_h = encoder_embed_h(encoder_inputs)
    state_h = encoder_embed_lambda(state_h)

    encoder_embed_c = keras.layers.Embedding(num_stocks + 1, layer_sizes[0])
    state_c = encoder_embed_c(encoder_inputs)
    state_c = encoder_embed_lambda(state_c)

    init_state = [state_h, state_c]

    decoder_input_layer = keras.layers.Masking(mask_value=0.,
                                               batch_input_shape=(batch_size, None, n_features))
    # decoder_input.set_weights(self.generalist.layers[0].get_weights())

    decoder_lstms = {}
    decoder_dropouts = {}
    for i, size in enumerate(layer_sizes):
        decoder_lstms[i] = keras.layers.LSTM(size, return_sequences=True,
                                             return_state=True,
                                             stateful=stateful)
        decoder_dropouts[i] = keras.layers.Dropout(0.2)

    decoder_dense = keras.layers.Dense(1, activation='linear')

    # decoder_dense.set_weights(self.generalist.layers[-1].get_weights())

    # encoder = keras.layers.LSTM(self.gen_dims[1], return_sequences=True, return_state=True)

    # encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    output = decoder_input_layer(decoder_inputs)
    # print(self.num_tags)


    for i in range(len(layer_sizes)):
        next_state = [state_h, state_c]
        if i == 0:
            output, state_h, state_c = decoder_lstms[i](output, initial_state=init_state)
        else:
            output, state_h, state_c = decoder_lstms[i](output, initial_state=next_state)
        output = decoder_dropouts[i](output)

    output = decoder_dense(output)

    model = keras.Model([decoder_inputs, encoder_inputs], output)

    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=keras.losses.MAE,
        metrics=[keras.metrics.MAPE]
    )

    copy_weights_layers = [4, 6, 7, 8, 9, 10, 11, 12]

    for i, layer_i in enumerate(copy_weights_layers):
        model.layers[layer_i].set_weights(gen_model.layers[i].get_weights())

    if(spes_model):
        model.set_weights(spes_model.get_weights())

    encoder_model = keras.Model(encoder_inputs, init_state)

    decoder_state_inputs = [keras.layers.Input(shape=(layer_sizes[0],)),
                            keras.layers.Input(shape=(layer_sizes[0],))]
    decoder_outputs = decoder_input_layer(decoder_inputs)

    for i in range(len(layer_sizes)):
        next_state = [state_h, state_c]
        if i == 0:
            decoder_outputs, state_h, state_c = decoder_lstms[i](decoder_outputs, initial_state=decoder_state_inputs)
        else:
            decoder_outputs, state_h, state_c = decoder_lstms[i](decoder_outputs, initial_state=next_state)
        decoder_outputs = decoder_dropouts[i](decoder_outputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = keras.Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

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

def main(mode='gen'):
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()

    # data = data[data['stock'] == 'AAPL']

    # feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
    feature_list = ['open', 'high', 'low', 'volume', 'price']

    n_features = len(feature_list)
    layer_sizes = [41, 41, 41]
    batch_size = 17
    model = build_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size, stateful=False)

    min_max_scaler_X = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    X = min_max_scaler_X.fit_transform(data[feature_list].values)
    y = min_max_scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    # y = data['next_price'].values.reshape(-1, 1)
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)


    X_no_pad, X, X_test = group_by_stock(X, n_features)
    y_no_pad, y, y_test = group_by_stock(y, 1)

    stock_id = 12
    if mode == 'gen':


        model.fit(X, y, batch_size=batch_size, epochs=300, shuffle=False)


        pred_model = build_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=1,stateful=True)


        pred_model.set_weights(model.get_weights())


        X_reshaped = X[stock_id].reshape(1, X[stock_id].shape[0], X[stock_id].shape[1])
        y_reshaped = y[stock_id].reshape(1, y[stock_id].shape[0], y[stock_id].shape[1])


        result_train = pred_model.predict(X_no_pad[stock_id].reshape(1, X_no_pad[stock_id].shape[0],
                                                                     X_no_pad[stock_id].shape[1]))
        result_train = result = min_max_scaler_y.inverse_transform(result_train[0]).reshape(-1)
        y_train = min_max_scaler_y.inverse_transform(y_no_pad[stock_id]).reshape(-1)



        result = pred_model.predict(X_test[stock_id].reshape(1, X_test[stock_id].shape[0], X_test[stock_id].shape[1]))
        # result = pred_model.predict(X[stock_id].reshape(1, X[stock_id].shape[0], X[stock_id].shape[1]))
        # result = pred_model.predict(X[stock_id].reshape(1, X[stock_id].shape[0], X[stock_id].shape[1]))[0].reshape(-1)

        result = min_max_scaler_y.inverse_transform(result[0]).reshape(-1)
        y_test = min_max_scaler_y.inverse_transform(y_test[stock_id]).reshape(-1)
        # y_test = min_max_scaler_y.inverse_transform(y[stock_id]).reshape(-1)
        # y_test = y[stock_id].reshape(-1)


        pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b')
        pd.DataFrame({'Actual': y_test})['Actual'].plot(label='Actual', c='r', linestyle='--')

        pd.DataFrame({'Predicted': result_train}).plot(label='Predicted', c='g')
        pd.DataFrame({'Actual': y_train})['Actual'].plot(label='Actual', c='y', linestyle='--')

        model.save_weights('model/gen_weights.h5')
        model.save('model/gen.h5')


    else:

        gen_model = keras.models.load_model('model/gen.h5')

        spes_model, _, _ = build_spes_network(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                        stateful=False, gen_model=gen_model, spes_model=None, batch_size=batch_size)


        X_stock_list = np.array([[[i]] for i in range(len(X))])
        spes_model.fit([X, X_stock_list], y, batch_size=batch_size, epochs=1, shuffle=False)

        spes_pred_model, encoder_model, decoder_model = build_spes_network(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                             stateful=True, gen_model=gen_model, spes_model=spes_model, batch_size=1)


        return
        result_train, = spes_pred_model.predict([X_no_pad[stock_id].reshape(1, X_no_pad[stock_id].shape[0],
                                                                           X_no_pad[stock_id].shape[1]),
                                                np.array([[[stock_id]]])])

        result_train = min_max_scaler_y.inverse_transform(result_train[0]).reshape(-1)
        y_train = min_max_scaler_y.inverse_transform(y_no_pad[stock_id]).reshape(-1)


        result, _, _ = spes_pred_model.predict([X_test[stock_id].reshape(1, X_test[stock_id].shape[0],
                                                                   X_test[stock_id].shape[1]),
                                          np.array([[[stock_id]]])])

        result = min_max_scaler_y.inverse_transform(result[0]).reshape(-1)
        y_test = min_max_scaler_y.inverse_transform(y_test[stock_id]).reshape(-1)

        pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b')
        pd.DataFrame({'Actual': y_test})['Actual'].plot(label='Actual', c='r', linestyle='--')
        pd.DataFrame({'Predicted': result_train}).plot(label='Predicted', c='g')
        pd.DataFrame({'Actual': y_train})['Actual'].plot(label='Actual', c='y', linestyle='--')

        print(spes_model.summary())



    plt.show()

main('spes')
