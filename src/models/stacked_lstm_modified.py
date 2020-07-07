import tensorflow as tf
from tensorflow import keras

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print('Using GPU')
    from tensorflow.python.keras.layers import CuDNNLSTM as LSTM
else:
    print('Using CPU')
    from tensorflow.keras.layers import LSTM

class StackedLSTM_Modified(keras.models.Model):
    def __init__(self,
                 n_features,
                 layer_sizes,
                 dropout=.2,
                 **_):
        X = tf.keras.layers.Input(shape=(None, n_features), name='X')

        output = X
        for i, size in enumerate(layer_sizes):
            lstm = LSTM(size, return_sequences=True, return_state=False)
            output = lstm(output)
            output = tf.keras.layers.Dropout(dropout)(output)

        next_price = tf.keras.layers.Dense(1, activation='linear')(output)
        direction = tf.keras.layers.Dense(1, activation='sigmoid')(output)
        super(StackedLSTM_Modified, self).__init__([X] , [next_price, direction],
                                          name='LSTM_stacked_modified')

