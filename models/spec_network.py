from keras import Model, Input
from keras.layers import Masking

from models.decoder import Decoder
from models.encoder import Encoder


class SpecializedNetwork(Model):
    def __init__(self, n_features, num_stocks, layer_sizes, stateful, batch_size):
        stock_id = Input(shape=(1, 1), name='Stock_ID')
        encoder = Encoder(num_stocks, layer_sizes[0])
        state_h, state_c = encoder(stock_id)

        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)
        decoder = Decoder(n_features, layer_sizes, stateful, batch_size)
        next_price, state_h, state_c = decoder([masked_X, state_h, state_c])

        super(SpecializedNetwork, self).__init__([X, stock_id], next_price, name='Specialized')
        self.encoder = encoder
        self.decoder = decoder
