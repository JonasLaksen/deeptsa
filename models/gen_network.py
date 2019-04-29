from keras import Input, Model
from keras.backend import zeros
from keras.layers import Masking, Lambda

from models.bidir_lstm import BidirLSTM
from models.decoder import Decoder
from models.stacked_lstm import StackedLSTM


class GeneralizedNetwork(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)

        init_h, init_c = Input(tensor=zeros((batch_size, layer_sizes[0]))), \
                         Input(tensor=zeros((batch_size, layer_sizes[0])))

        decoder = Decoder(n_features, layer_sizes, stateful, batch_size)
        next_price, _, _ = decoder([masked_X, init_h, init_c])
        super(GeneralizedNetwork, self).__init__([X, init_h, init_c], next_price, name='Generalized')
