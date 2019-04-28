from keras import Model, Input
from keras.layers import LSTM, Dense, Dropout


class StackedLSTM(Model):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')
        init_h, init_c = Input(shape=(layer_sizes[0],), name='Initial_h'), \
                         Input(shape=(layer_sizes[0],), name='Initial_c')
        state_h, state_c = init_h, init_c

        output = X
        for size in layer_sizes:
            lstm = LSTM(size, return_sequences=True, return_state=True, stateful=stateful)
            output, state_h, state_c = lstm(output, initial_state=[state_h, state_c])
            output = Dropout(.4)(output)

        next_price = Dense(1, activation='linear')(output)
        super(StackedLSTM, self).__init__([X, init_h, init_c], [next_price, state_h, state_c], name='StackedLSTM')
