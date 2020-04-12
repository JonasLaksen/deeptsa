from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Reshape, Embedding


class Encoder(Model):
    def __init__(self, num_stocks, cell_size, n_states = 2):
        encoder_inputs = Input(shape=(1, 1), name='Stock_ID')

        states = []
        for i in range(n_states):
            state = Embedding(num_stocks + 1, cell_size)(encoder_inputs)
            state = Reshape((cell_size,), name='State_{}'.format(i))(state)
            states.append(state)

        super(Encoder, self).__init__(encoder_inputs, states, name='Encoder')
