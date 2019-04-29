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

        n_states = 4
        zero_states = [Input(tensor=zeros((batch_size, layer_sizes[0]))) for _ in range(n_states)]

        # lstm = StackedLSTM(n_features, layer_sizes, stateful, batch_size)
        lstm = BidirLSTM(n_features, layer_sizes, stateful, batch_size)
        next_price, *_ = lstm([masked_X] + zero_states)
        super(GeneralizedNetwork, self).__init__([X] + zero_states, next_price, name='Generalized')
