from keras import Model, Input

from models.encoder import Encoder


class SpecializedNetwork(Model):
    def __init__(self, n_features, num_stocks, layer_sizes, decoder, return_states=False, is_bidir=False):
        stock_id = Input(shape=(1, 1), name='Stock_ID')
        encoder = Encoder(num_stocks, layer_sizes[0], n_states=4*len(layer_sizes) if is_bidir else 2*len(layer_sizes))
        init_states = encoder(stock_id)

        X = Input(shape=(None, n_features), name='X')
        next_price, *states = decoder([X] + init_states)

        super(SpecializedNetwork, self).__init__([X, stock_id],
                                                 [next_price] + (states if return_states else []),
                                                 name='Specialized')
        self.decoder = decoder
        self.encoder = encoder
