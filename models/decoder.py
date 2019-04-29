from keras import Model, Input
from keras.backend import zeros

from models.bidir_lstm import BidirLSTM
from models.stacked_lstm import StackedLSTM


class Decoder(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')

        n_states = 4
        init_states = [Input(tensor=zeros((batch_size, layer_sizes[0]))) for _ in range(n_states)]

        lstm = BidirLSTM(n_features, layer_sizes, stateful, batch_size)
        # lstm = StackedLSTM(n_features, layer_sizes, stateful, batch_size)
        next_price, *states = lstm([X] + init_states)
        super(Decoder, self).__init__([X] + init_states, [next_price] + states, name='Decoder')
