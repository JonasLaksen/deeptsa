from tensorflow import keras


class Encoder(keras.models.Model):
    def __init__(self, num_stocks, cell_size, n_states = 2):
        encoder_inputs = keras.layers.Input(shape=(1, 1), name='Stock_ID')

        states = []
        for i in range(n_states):
            state = keras.layers.Embedding(num_stocks + 1, cell_size)(encoder_inputs)
            state = keras.layers.Reshape((cell_size,), name='State_{}'.format(i))(state)
            states.append(state)

        super(Encoder, self).__init__(encoder_inputs, states, name='Encoder')
