from keras import Model, Input
from keras.layers import LSTM, Dense, Dropout, Masking


class StackedLSTM(Model):
    def __init__(self, n_features, layer_sizes, batch_size, init_all_layers=True, return_states=True):
        X = Input(batch_shape=(batch_size, None, n_features), name='X')

        init_states = [Input(shape=(layer_sizes[0],), name='State_{}'.format(i)) for i in range(len(layer_sizes) * 2)]
        new_states = []
        states = []

        output = X
        for i, size in enumerate(layer_sizes):
            lstm = LSTM(size, return_sequences=True, return_state=True)
            # If init_all_layers, set initial state of all layers equal the input init_states
            # Else, set the initial state of the first layer equals the first two states of init_states and the other
            # layers init_states equal the end state of previous layers
            output, *states = lstm(output, initial_state=init_states[i * 2:(i * 2) + 2] if (
                    init_all_layers or i == 0) else states)
            new_states = new_states + states
            output = Dropout(.4)(output)

        next_price = Dense(1, activation='linear')(output)
        super(StackedLSTM, self).__init__([X] + init_states, [next_price] + (new_states if return_states else []),
                                          name='LSTM_stacked')
