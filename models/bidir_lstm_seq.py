import keras
from keras.layers import Bidirectional, LSTM, Dense, Masking, Dropout, Average, Input
from keras import Model

def build_model(n_features, layer_sizes, batch_size, return_states=True):

    layer_size = layer_sizes[0]
    input_lstm = Input(batch_shape=(batch_size, None, n_features))

    init_states = [Input(shape=(layer_size,), name='State_{}'.format(i)) for i in range(4)]

    # masking_input = Masking(mask_value=0.0)
    # input_lstm = masking_input(input_lstm)

    left_masking = Masking(mask_value=0.0)
    left_lstm_1 = LSTM(layer_size, return_sequences=True, return_state=True)
    left_dropout_1 = Dropout(0.2)
    # left_lstm_2 = LSTM(layer_size, return_sequences=True, return_state=True, stateful=stateful)
    # left_dropout_2 = Dropout(0.2)
    left_output = left_masking(input_lstm)
    left_output, *left_states = left_lstm_1(left_output, initial_state=init_states[:2])
    left_output = left_dropout_1(left_output)
    # left_output, *left_states = left_lstm_2(left_output, initial_state=left_states)
    # left_output = left_dropout_2(left_output)

    right_masking = Masking(mask_value=0.0)
    right_lstm_1 = LSTM(layer_size, return_sequences=True, return_state=True, go_backwards=True)
    right_dropout_1 = Dropout(0.2)
    # right_lstm_2 = LSTM(layer_size, return_sequences=True, return_state=True, stateful=stateful, go_backwards=True)
    # right_dropout_2 = Dropout(0.2)
    right_output = right_masking(input_lstm)
    right_output, *right_states = right_lstm_1(right_output, initial_state=init_states[2:])
    right_output = right_dropout_1(right_output)
    # right_output, *right_states = right_lstm_2(right_output, initial_state=right_states)
    # right_output = right_dropout_2(right_output)

    merge_layer = Average()

    next_price = merge_layer([left_output, right_output])

    output_layer = Dense(1, activation='linear')

    next_price = output_layer(next_price)
    new_states = left_states + right_states

    model = Model([input_lstm] + init_states, [next_price] + (new_states if return_states else []))

    return model
