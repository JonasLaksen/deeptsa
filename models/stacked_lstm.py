from keras import Model, Input
from keras.layers import LSTM, Dense, Dropout


class StackedLSTM(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')

        # This model needs two initial states:
        init_states = [Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(2)]
        states = init_states

        output = X
        for size in layer_sizes:
            lstm = LSTM(size, return_sequences=True, return_state=True, stateful=stateful)
            output, *states = lstm(output, initial_state=states)
            output = Dropout(.4)(output)

        next_price = Dense(1, activation='linear')(output)
        super(StackedLSTM, self).__init__([X] + init_states, [next_price] + states, name='LSTM')
