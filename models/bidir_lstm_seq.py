from keras.layers import Dense, Dropout, Average, Input
from keras import backend as K, Model

if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
else:
    from keras.layers import LSTM


def build_model(n_features, layer_sizes, return_states=True, dropout=1.):
    layer_size = layer_sizes[0]
    input_lstm = Input(shape=(None, n_features))

    init_states = [Input(shape=(layer_size,), name='State_{}'.format(i)) for i in range(4)]

    left_lstm_1 = LSTM(layer_size, return_sequences=True, return_state=True)
    left_dropout_1 = Dropout(dropout)
    left_output, *left_states = left_lstm_1(input_lstm, initial_state=init_states[:2])
    left_output = left_dropout_1(left_output)

    right_lstm_1 = LSTM(layer_size, return_sequences=True, return_state=True, go_backwards=True)
    right_dropout_1 = Dropout(dropout)
    right_output, *right_states = right_lstm_1(input_lstm, initial_state=init_states[2:])
    right_output = right_dropout_1(right_output)

    merge_layer = Average()
    next_price = merge_layer([left_output, right_output])

    output_layer = Dense(1, activation='linear')

    next_price = output_layer(next_price)
    new_states = left_states + right_states

    model = Model([input_lstm] + init_states, [next_price] + (new_states if return_states else []))

    return model
