from tensorflow import keras

from src.models.encoder import Encoder


class SpecializedNetwork(keras.models.Model):
    def __init__(self, n_features, num_stocks, layer_sizes, decoder, return_states=False):
        stock_id = keras.layers.Input(shape=(1, 1), name='Stock_ID')
        encoder = Encoder(num_stocks, layer_sizes[0], n_states=2*len(layer_sizes))
        init_states = encoder(stock_id)

        X = keras.layers.Input(shape=(None, n_features), name='X')
        next_price = decoder([X] + init_states)

        super(SpecializedNetwork, self).__init__([X, stock_id],
                                                 [next_price],
                                                 name='Specialized')
        self.decoder = decoder
        self.encoder = encoder
