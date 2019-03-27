import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from constants import stock_list
import numpy as np

def build_network(n_features, layer_sizes, stateful, batch_size, loss='mse'):

    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
    for i, layer_size in enumerate(layer_sizes):
        if i==0:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful))
        else:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=loss,
    )

    return model

def group_by_stock(data, n_features):
    group_by_dict = {}
    for row in data:
        try:
            group_by_dict[row[0]].append(row[1:])
        except:
            group_by_dict[row[0]] = [row[1:]]

    data = list(map(lambda x: np.array(group_by_dict[x])[:-50], group_by_dict.keys()))
    test_data = list(map(lambda x: np.array(group_by_dict[x])[-50:], group_by_dict.keys()))


    max_length = np.max([x.shape[0] for x in data])
    for i in range(len(data)):
        data[i] = np.append(data[i], np.zeros((max_length - data[i].shape[0], n_features)), axis=0)

    return np.array(data), np.array(test_data)

if __name__ == '__main__':
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()

    # data = data[data['stock'] == 'AAPL']

    # feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
    feature_list = ['open', 'high', 'low', 'price']

    n_features = len(feature_list)
    layer_sizes = [23, 23, 23]
    batch_size = 17
    model = build_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size, stateful=False)

    min_max_scaler_X = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    X = min_max_scaler_X.fit_transform(data[feature_list].values)
    y = min_max_scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    # y = data['next_price'].values.reshape(-1, 1)
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)


    X, X_test = group_by_stock(X, n_features)
    y, y_test = group_by_stock(y, 1)


    model.fit(X, y, batch_size=batch_size, epochs=200, shuffle=False)

    print(model.summary())

    pred_model = build_network(n_features=n_features, layer_sizes=layer_sizes, batch_size=1,stateful=True)


    pred_model.set_weights(model.get_weights())

    stock_id = 12

    X_reshaped = X[stock_id].reshape(1, X[stock_id].shape[0], X[stock_id].shape[1])
    y_reshaped = y[stock_id].reshape(1, y[stock_id].shape[0], y[stock_id].shape[1])

    # for i in range(2):
    #     pred_model.reset_states()
    #     pred_model.fit(X_reshaped, y_reshaped, batch_size=1, epochs=1, shuffle=False)

    result_train = pred_model.predict(X[stock_id].reshape(1, X[stock_id].shape[0], X[stock_id].shape[1]))
    result_train = result = min_max_scaler_y.inverse_transform(result_train[0]).reshape(-1)
    y_train = min_max_scaler_y.inverse_transform(y[stock_id]).reshape(-1)



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


    plt.show()


