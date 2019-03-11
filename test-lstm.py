import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn import preprocessing

def build_network(n_features, layer_sizes):

    model = keras.Sequential()

    for i, layer_size in enumerate(layer_sizes):
        if i==0:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True, batch_input_shape=(1, None, n_features)))
        else:
            model.add(keras.layers.LSTM(layer_size, return_sequences=True))
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

    feature_list = ['positive', 'negative', 'neutral', 'trendscore']

    n_features = len(feature_list)
    layer_sizes = [11, 11]
    model = build_network(n_features=n_features, layer_sizes=layer_sizes)

    min_max_scaler_X = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    X = min_max_scaler_X.fit_transform(data[feature_list].values)
    y = min_max_scaler_y.fit_transform(data['price'].values.reshape(-1, 1))

    print(X)
    print(y)

    model.fit(X.reshape(1, X.shape[0], X.shape[1]), y.reshape(1, len(y), 1), batch_size=1, epochs=500, shuffle=False)
    print(model.summary())



