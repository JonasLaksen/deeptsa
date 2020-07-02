import tensorflow as tf
from tensorflow import keras


if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('Using GPU')
    from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
else:
    print('Using CPU')
    from tensorflow.keras.layers import LSTM

class BidirLSTM(keras.models.Model):
    def __init__(self, n_features, layer_sizes, return_states=True, dropout=.2, **_):
        X = tf.keras.layers.Input(shape=(None, n_features), name='X')
        #masking = tf.keras.layers.Masking(mask_value=-1.0)(X)

        output = X
        #output = masking
        for i, size in enumerate(layer_sizes):
            lstm = tf.keras.layers.Bidirectional( LSTM(size, return_sequences=True, return_state=True), merge_mode='ave', dtype='float64')
            output, *states = lstm(output)
            output = tf.keras.layers.Dropout(dropout)(output)

        next_price = tf.keras.layers.Dense(1, activation='linear')(output)
        super(BidirLSTM, self).__init__([X], [next_price], name='LSTM_bidir')

