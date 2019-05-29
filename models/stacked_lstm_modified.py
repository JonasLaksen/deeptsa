from keras import Model, Input
from keras import backend as K
from keras.layers import Dense, Dropout

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
else:
    from keras.layers import LSTM


class StackedLSTM_Modified(Model):
    def __init__(self, n_features, layer_sizes, return_states=True, dropout=.2, **_):
        X = Input(shape=(None, n_features), name='X')
        init_states = [Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(len(layer_sizes) * 2)]
        new_states = []

        output = X
        for i, size in enumerate(layer_sizes):
            lstm = LSTM(size, return_sequences=True, return_state=True)
            output, *states = lstm(output, initial_state=init_states[i * 2:(i * 2) + 2])
            new_states = new_states + states
            output = Dropout(dropout)(output)

        next_price = Dense(1, activation='linear')(output)
        direction = Dense(1, activation='sigmoid')(output)
        super(StackedLSTM_Modified, self).__init__([X] + init_states, [next_price, direction] + (new_states if return_states else []),
                                          name='LSTM_stacked_modified')
