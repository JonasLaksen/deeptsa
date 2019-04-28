from keras import Input, Model
from keras.layers import Masking

from models.bidir_lstm import BidirLSTM
from models.stacked_lstm import StackedLSTM


class GeneralizedNetwork(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        masked_X = Masking(mask_value=0., batch_input_shape=(batch_size, None, n_features), name='Masked_X')(X)
        init_h, init_c = Input(shape=(layer_sizes[0],), name='Initial_h'), \
                         Input(shape=(layer_sizes[0],), name='Initial_c')

        # stackedLSTM = StackedLSTM(n_features, layer_sizes, stateful, batch_size)
        bidirLSTM = BidirLSTM(n_features, layer_sizes, stateful, batch_size)
        next_price, _, _ = bidirLSTM([masked_X, init_h, init_c])
        super(GeneralizedNetwork, self).__init__([X, init_h, init_c], next_price, name='Generalized')
