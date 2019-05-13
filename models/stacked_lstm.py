from keras import Model, Input
from keras import backend as K
from keras.layers import Dense, Dropout, Masking

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
else:
    from keras.layers import LSTM


class StackedLSTM(Model):
    def __init__(self, n_features, layer_sizes, batch_size, return_states=True):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)

        init_states = [Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(len(layer_sizes) * 2)]
        new_states = []

        output = masked_X
        for i, size in enumerate(layer_sizes):
            lstm = LSTM(size, return_sequences=True, return_state=True)
            output, *states = lstm(output, initial_state=init_states[i * 2:(i * 2) + 2])
            new_states = new_states + states
            output = Dropout(.4)(output)

        next_price = Dense(1, activation='linear')(output)
        super(StackedLSTM, self).__init__([X] + init_states, [next_price] + (new_states if return_states else []),
                                          name='LSTM_stacked')
