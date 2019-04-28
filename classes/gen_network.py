from keras import Sequential
from keras.layers import Masking, LSTM, Dense


class GeneralizedNetwork(Sequential):
    def __init__(self, n_features, layer_sizes, stateful, batch_size):
        super(GeneralizedNetwork, self).__init__()

        self.add(Masking(mask_value=0.0, batch_input_shape=(batch_size, None, n_features)))
        for i, layer_size in enumerate(layer_sizes):
            self.add(LSTM(layer_size, return_sequences=True, stateful=stateful, dropout=.2))

        self.add(Dense(1, activation='linear'))
