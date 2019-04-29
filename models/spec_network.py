from keras import Model, Input
from keras.layers import Masking

from models.bidir_lstm import BidirLSTM
from models.encoder import Encoder
from models.stacked_lstm import StackedLSTM


class SpecializedNetwork(Model):
    def __init__(self, n_features, num_stocks, layer_sizes, stateful, batch_size):
        stock_id = Input(shape=(1, 1), name='Stock_ID')
        encoder = Encoder(num_stocks, layer_sizes[0], n_states=4)
        init_states = encoder(stock_id)

        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)
        decoder = BidirLSTM(n_features, layer_sizes, stateful, batch_size)
        next_price, *_ = decoder([masked_X] + init_states)

        super(SpecializedNetwork, self).__init__([X, stock_id], next_price, name='Specialized')
        self.encoder = encoder
        self.decoder = decoder
