import tensorflow as tf
from tensorflow import keras

def build_network(n_features, layer_sizes):

    model = keras.Sequential()

    for layer_size in layer_sizes:
        model.add(keras.layers.LSTM(layer_sizes, return_sequences=True))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
    )

    return model

if __name__ == '__main__':
    n_features = 4
    layer_sizes = [11, 11, 11]
    model = build_network(n_features=n_features, layer_sizes=layer_sizes)

    print(model.summary())

