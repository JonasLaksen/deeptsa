import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt


def build_network(n_features, layer_sizes, stateful):

    model = keras.Sequential()

    for i, layer_size in enumerate(layer_sizes):
        if i==0:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful,batch_input_shape=(1, None, n_features)))
        else:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True, stateful=stateful))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
    )

    return model

if __name__ == '__main__':


    data = pd.read_csv('dataset.csv', index_col=0)

    data = data[data['stock'] == 'AAPL']

    feature_list = ['prev_price_0', 'positive', 'negative', 'neutral', 'trendscore']

    n_features = len(feature_list)
    layer_sizes = [11]
    model = build_network(n_features=n_features, layer_sizes=layer_sizes, stateful=False)



    min_max_scaler_X = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    X = min_max_scaler_X.fit_transform(data[feature_list].values)
    y = min_max_scaler_y.fit_transform(data['price'].values.reshape(-1, 1))

    X_test = X[-5:]
    y_test = y[-5:]
    X = X[:-5]
    y = y[:-5]



    model.fit(X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1), batch_size=1, epochs=300, shuffle=False)

    pred_model = build_network(n_features=n_features, layer_sizes = layer_sizes, stateful=True)

    pred_model.set_weights(model.get_weights())

    pred_model.fit(X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1), batch_size=1, epochs=1, shuffle=False)

    result = pred_model.predict(X.reshape(1, X.shape[0], X.shape[1]))
    print(result)
    print(y)

    result = min_max_scaler_y.inverse_transform(result[0]).reshape(-1)
    y_test = min_max_scaler_y.inverse_transform(y).reshape(-1)

    pd.DataFrame({'Predicted': result}).plot(label='Predicted', c='b')
    pd.DataFrame({'Actual': y_test})['Actual'].plot(label='Actual', c='r', linestyle='--')

    plt.show()

    print(model.summary())



