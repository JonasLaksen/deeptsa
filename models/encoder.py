from keras import Model, Input
from keras.layers import Reshape, Embedding


class Encoder(Model):
    def __init__(self, num_stocks, cell_size):
        encoder_inputs = Input(shape=(1, 1), name='Stock_ID')

        init_h = Embedding(num_stocks + 1, cell_size)(encoder_inputs)
        init_h = Reshape((cell_size,), name='Initial_h')(init_h)

        init_c = Embedding(num_stocks + 1, cell_size)(encoder_inputs)
        init_c = Reshape((cell_size,), name='Initial_c')(init_c)

        super(Encoder, self).__init__(encoder_inputs, [init_h, init_c], name='Encoder')
