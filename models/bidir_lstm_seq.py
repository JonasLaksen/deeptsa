from keras import backend as K, Model
from keras.layers import Dense, Dropout, Concatenate, Input

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
else:
    from keras.layers import LSTM


def build_model(n_features, layer_sizes, return_states=True, dropout=1.):
    X = Input(shape=(None, n_features))

    init_states = [Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(4 * len(layer_sizes))]
    new_states = []
    output = X

    for i, layer_size in enumerate(layer_sizes):
        left_lstm = LSTM(layer_size, return_sequences=True, return_state=True)
        left_output, *left_states = left_lstm(output, initial_state=init_states[i * 4: (i * 4) + 2])
        left_output = Dropout(dropout)(left_output)

        right_lstm = LSTM(layer_size, return_sequences=True, return_state=True, go_backwards=True)
        right_output, *right_states = right_lstm(output, initial_state=init_states[i * 4 + 2: (i * 4) + 4])
        right_output = Dropout(dropout)(right_output)

        output = Concatenate()([left_output, right_output])
        new_states = new_states + left_states + right_states

    next_price = Dense(1, activation='linear')(output)

    return Model([X] + init_states, [next_price] + (new_states if return_states else []))
