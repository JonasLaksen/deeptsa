from keras import Model, Input

from classes.decoder import Decoder
from classes.encoder import Encoder


class SpecializedNetwork(Model):
    def __init__(self, n_features, num_stocks, layer_sizes, stateful, batch_size, gen_model):
        stock_id = Input(shape=(1, 1))
        encoder = Encoder(num_stocks, layer_sizes[0])
        state_h, state_c = encoder(stock_id)

        X = Input(batch_shape=(batch_size, None, n_features))
        decoder = Decoder(n_features, layer_sizes, stateful, batch_size)
        next_price, state_h, state_c = decoder([X, state_h, state_c])

        copy_weights_layers = [4, 5, 6, 7, 8, 9, 10]
        for i, layer_i in enumerate(copy_weights_layers):
            decoder.layers[layer_i].set_weights(gen_model.layers[i + 1].get_weights())

        super(SpecializedNetwork, self).__init__([X, stock_id], next_price, name='Specialized')
        self.encoder = encoder
        self.decoder = decoder
