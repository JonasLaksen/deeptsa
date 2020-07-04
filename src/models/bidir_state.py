import tensorflow as tf
from tensorflow import keras


if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('Using GPU')
    from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
else:
    print('Using CPU')
    from tensorflow.keras.layers import LSTM

class BidirLSTMWithState(keras.models.Model):
    def __init__(self, n_features, layer_sizes, return_states=True, dropout=.2, **_):
        X = tf.keras.layers.Input(shape=(None, n_features), name='X')
        init_states = [tf.keras.layers.Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(len(layer_sizes) * 4)]
        new_states = []

        output = X
        for i, size in enumerate(layer_sizes):
            lstm = tf.keras.layers.Bidirectional( LSTM(size, return_sequences=True, return_state=True), merge_mode='ave')
            output, *states = lstm(output, initial_state=init_states[i*4:(i*4)+4])
            new_states = new_states + states
            output = tf.keras.layers.Dropout(dropout)(output)

        next_price = tf.keras.layers.Dense(1, activation='linear')(output)
        super(BidirLSTMWithState, self).__init__([X] + init_states, [next_price], name='LSTM_bidir_state')

