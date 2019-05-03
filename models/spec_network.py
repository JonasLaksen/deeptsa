from keras import Model, Input
from keras.backend import zeros
from keras.layers import Masking

from models.encoder import Encoder
from models.stacked_lstm import StackedLSTM


class SpecializedNetwork(Model):
    def __init__(self, n_features, num_stocks, layer_sizes, batch_size, init_all_layers=True, return_states=False):
        stock_id = Input(shape=(1, 1), name='Stock_ID')
        encoder = Encoder(num_stocks, layer_sizes[0], n_states=len(layer_sizes) * 2 if init_all_layers else 2)
        init_states = encoder(stock_id)

        zero_states = [Input(tensor=zeros((batch_size, layer_sizes[0]))) for _ in range((len(layer_sizes) * 2) - 2)]

        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        # masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)
        decoder = StackedLSTM(n_features, layer_sizes, batch_size, init_all_layers=init_all_layers)
        next_price, *states = decoder([X] + init_states + (zero_states if not init_all_layers else []))

        super(SpecializedNetwork, self).__init__([X, stock_id] + (zero_states if not init_all_layers else []),
                                                 [next_price] + (states if return_states else []),
                                                 name='Specialized')
        self.decoder = decoder
        self.encoder = encoder
