import tensorflow as tf
from tensorflow import keras

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('Using GPU')
    from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
else:
    print('Using CPU')
    from tensorflow.keras.layers import LSTM


class StackedLSTMWithState(keras.models.Model):
    def __init__(self, n_features, layer_sizes, dropout=.2, **_):
        X = tf.keras.layers.Input(shape=(None, n_features), name='X')
        init_states = [tf.keras.layers.Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(len(layer_sizes) * 2)]

        output = X
        for i, size in enumerate(layer_sizes):
            lstm = LSTM(size, return_sequences=True, return_state=False)
            output = lstm(output, initial_state=init_states[i * 2:(i * 2) + 2])
            # output = tf.keras.layers.Dropout(dropout)(output)

        next_price = tf.keras.layers.Dense(1, activation='linear')(output)
        super(StackedLSTMWithState, self).__init__([X] + init_states, [next_price], name='LSTM_stacked')