import numpy as np
import tensorflow as tf

tf.random.set_seed(0)
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint

from src.models.spec_network import SpecializedNetwork
from src.models.stacked_lstm import StackedLSTM
from src.utils import evaluate, plot


class LSTMOneOutput:
    def __init__(self, X_train, y_train, X_val, y_val, model_generator, dropout, optimizer, loss, stock_list, seed,
                 feature_list,
                 n_features, batch_size, layer_sizes, X_stocks) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model_generator = model_generator
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.stock_list = stock_list
        self.loss = loss
        self.optimizer = optimizer
        self.seed = seed
        self.feature_list = feature_list
        self.n_features = n_features
        self.batch_size = batch_size
        self.X_stocks = X_stocks
        self.gen_model = self.model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=False,
                                              dropout=self.dropout)

        self.gen_model.compile(optimizer=self.optimizer, loss=self.loss)

        self.decoder = self.model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=True,
                                            dropout=self.dropout)

        self.is_bidir = self.model_generator is not StackedLSTM
        # self.spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(stock_list), layer_sizes=layer_sizes,
        #                                      decoder=self.decoder, is_bidir=self.is_bidir)
        # self.spec_model.compile(optimizer=self.optimizer, loss=self.loss)
        super().__init__()

    def meta(self, description, epochs):
        dict = {
            'dropout': self.dropout,
            'epochs': epochs,
            'time': str(datetime.now()),
            'features': ', '.join(self.feature_list),
            'model-type': {'bidir' if self.is_bidir else 'stacked'},
            'layer-sizes': f"[{', '.join(str(x) for x in self.layer_sizes)}]",
            'loss': self.loss,
            'seed': self.seed,
            'description': description,
            'X-train-shape': self.X_train.shape,
            'X-val-shape': self.X_val.shape,
            'X-stocks': self.X_stocks
        }
        return '\n'.join([f'{key}: {value}' for (key, value) in dict.items()])

    def __str__(self):
        # To decide where to save data
        return f"{datetime.now()}"

    def train(self, gen_epochs, spech_epochs, copy_weights_from_gen_to_spec, load_spec,
              load_gen, train_general, train_specialized):
        print(f"Training on {self.X_train.shape[0]} stocks")
        losses = {}
        if load_gen:
            self.gen_model.load_weights(f'weights/{str(self.gen_model)}')
            print('Loaded general model')

        if train_general:
            general_loss, general_val_loss = self.train_general(gen_epochs, self.n_features, self.batch_size)
            losses['general_loss'] = general_loss
            losses['general_val_loss'] = general_val_loss

        if copy_weights_from_gen_to_spec:
            self.decoder.set_weights(self.gen_model.get_weights())

        if load_spec:
            self.spec_model.load_weights('weights/spec.h5')
            print('Loaded specialised model')

        if train_specialized:
            spec_loss, spec_val_loss = self.train_spec(self.X_train, self.y_train, self.X_val, self.y_val, spech_epochs,
                                                       self.n_features, self.batch_size)
            losses['spec_loss'] = spec_loss
            losses['spec_val_loss'] = spec_val_loss
        return losses

    def train_general(self, epochs, n_features, batch_size):
        is_bidir = self.model_generator is not StackedLSTM
        zero_states = [np.zeros((batch_size, self.layer_sizes[0]))] * len(self.layer_sizes) * 2 * (2 if is_bidir else 1)
        history = self.gen_model.fit([self.X_train] + zero_states, self.y_train,
                                     validation_data=([self.X_val] + zero_states, self.y_val),
                                     epochs=epochs,
                                     verbose=1,
                                     shuffle=False,
                                     batch_size=batch_size)
        # gen_model.load_weights("best-weights.hdf5")

        self.gen_pred_model = self.model_generator(n_features=n_features, layer_sizes=self.layer_sizes,
                                                   return_states=True,
                                                   dropout=self.dropout)
        self.gen_pred_model.set_weights(self.gen_model.get_weights())
        return history.history['loss'], history.history['val_loss']

    def train_spec(self, epochs, n_features, batch_size):
        is_bidir = self.model_generator is not StackedLSTM
        # Create the context model, set the decoder = the gen model
        history = self.spec_model.fit([self.X_train] + self.stock_list, self.y_train,
                                      validation_data=([self.X_val] + self.stock_list, self.y_val),
                                      batch_size=batch_size, epochs=epochs, shuffle=False,
                                      callbacks=[ModelCheckpoint('weights/spec.h5', period=1, save_weights_only=True)])
        # write_to_csv(f'plot_data/spec/loss/{filename}.csv', history.history)
        spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(self.X_train),
                                             layer_sizes=self.layer_sizes,
                                             return_states=True, decoder=self.spec_model.decoder, is_bidir=is_bidir)
        spec_pred_model.set_weights(self.spec_model.get_weights())
        return history.history['loss'], history.history['val_loss']

    def generate_general_model_results(self, scaler_y, y_type):
        print(str(self))
        model = self.gen_pred_model
        zero_states = [np.zeros((self.batch_size, self.layer_sizes[0]))] * len(self.layer_sizes) * 2 * (
            2 if self.is_bidir else 1)
        init_state = zero_states

        result_train, *new_states = model.predict([self.X_train] + init_state)
        # result_train = result_train[:, 10:]
        result_val = None
        for i in range(self.X_val.shape[1]):
            temp, *new_states = model.predict([np.append(self.X_train, self.X_val[:, :i + 1], axis=1)] + new_states)
            if result_val is None:
                result_val = temp[:, -1:]
            else:
                result_val = np.append(result_val, temp[:, -1:], axis=1)

        result_train, result_val, y_train_inv, y_val_inv = map(
            lambda x: np.array(list(map(scaler_y.inverse_transform, x))),
            [result_train, result_val, self.y_train, self.y_val])

        # y_train_inv = y_train_inv[:, 10:]
        val_evaluation = evaluate(result_val, y_val_inv, y_type)
        train_evaluation = evaluate(result_train, y_train_inv, y_type)
        print('Val: ', val_evaluation)
        print('Training:', train_evaluation)
        plot('Training', np.array(self.stock_list).reshape(-1), result_train[:3], y_train_inv[:3])
        plot('Val', np.array(self.stock_list).reshape(-1), result_val[:3], y_val_inv[:3])
        return { 'training': train_evaluation, 'validation':val_evaluation}
