from keras import Model, Input
from keras.layers import Masking, LSTM, Dense, Dropout


class Decoder(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features))
        h_input, c_input = Input(shape=(layer_sizes[0],)), Input(shape=(layer_sizes[0],))

        output = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features))(X)
        state_h, state_c = h_input, c_input

        for size in layer_sizes:
            lstm = LSTM(size, return_sequences=True, return_state=True, stateful=stateful)
            output, state_h, state_c = lstm(output, initial_state=[state_h, state_c])
            output = Dropout(.4)(output)

        next_price = Dense(1, activation='linear')(output)
        super(Decoder, self).__init__([X, h_input, c_input], [next_price, state_h, state_c], name='Decoder')
