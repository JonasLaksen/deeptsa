from keras import Model, Input

from models.stacked_lstm import StackedLSTM


class Decoder(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        h_input, c_input = Input(shape=(layer_sizes[0],), name='State_h'), \
                           Input(shape=(layer_sizes[0],), name='State_c')
        stackedLSTM = StackedLSTM(n_features, layer_sizes, stateful, batch_size)
        next_price, state_h, state_c = stackedLSTM([X, h_input, c_input])
        super(Decoder, self).__init__([X, h_input, c_input], [next_price, state_h, state_c], name='Decoder')
