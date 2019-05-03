import keras
from keras.layers import Bidirectional, LSTM, Dense, Masking, Dropout, Average, Input
from keras import Model

def build_model(n_features, layer_size, stateful, batch_size):

    input_lstm = Input(batch_shape=(batch_size, None, n_features))


    left = keras.Sequential()
    left.add(Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
    left.add(LSTM(layer_size, return_sequences=True, stateful=stateful))
    left.add(Dropout(0.2))

    left_input = left(input_lstm)

    right = keras.Sequential()
    right.add(Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
    right.add(LSTM(layer_size, return_sequences=True, stateful=stateful, go_backwards=True))
    right.add(Dropout(0.2))

    right_input = right(input_lstm)


    merge_layer = Average()

    output = merge_layer([left_input, right_input])

    output_layer = Dense(1, activation='linear')

    output = output_layer(output)

    model = Model(input_lstm, output)

    return model
