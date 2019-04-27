from keras import Model, Input
from keras.layers import Reshape, Embedding


class Encoder(Model):
    def __init__(self, num_stocks, cell_size):
        encoder_inputs = Input(shape=(1, 1))

        encoder_reshape = Reshape((cell_size,))
        init_h = Embedding(num_stocks + 1, cell_size)(encoder_inputs)
        init_h = encoder_reshape(init_h)

        init_c = Embedding(num_stocks + 1, cell_size)(encoder_inputs)
        init_c = encoder_reshape(init_c)

        super(Encoder, self).__init__(encoder_inputs, [init_h, init_c])
