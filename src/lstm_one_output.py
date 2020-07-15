import numpy as np
import tensorflow as tf

from src.models.bidir import BidirLSTM
from src.models.spec_network import SpecializedNetwork

tf.random.set_seed(0)
from datetime import datetime

from src.models.stacked_lstm import StackedLSTM
from src.utils import evaluate, plot_one


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

        self.gen_model.compile(optimizer=self.optimizer, loss=self.loss,
                               metrics=['mape', 'mae', 'mse', 'binary_accuracy'])

        self.decoder = self.model_generator(n_features=n_features, layer_sizes=layer_sizes, return_states=True,
                                            dropout=self.dropout)

        self.is_bidir = self.model_generator is BidirLSTM
        # self.spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(stock_list), layer_sizes=layer_sizes,
        #                                      decoder=self.decoder)
        # self.spec_model.compile(optimizer=self.optimizer, loss=self.loss)
        super().__init__()

    def meta(self, epochs):
        dict = {
            'dropout': self.dropout,
            'epochs': epochs,
            'time': str(datetime.now()),
            'features': ', '.join(self.feature_list),
            'model-type': 'bidir' if self.is_bidir else 'stacked',
            'layer-sizes': f"[{', '.join(str(x) for x in self.layer_sizes)}]",
            'loss': self.loss,
            'seed': self.seed,
            'X-train-shape': list(self.X_train.shape),
            'X-val-shape': list(self.X_val.shape),
            'y-train-shape': list(self.y_train.shape),
            'y-val-shape': list(self.y_val.shape),
            'X-stocks': list(self.X_stocks)
        }
        return dict

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
            spec_loss, spec_val_loss = self.train_spec(spech_epochs)
            losses['spec_loss'] = spec_loss
            losses['spec_val_loss'] = spec_val_loss
        return losses

    def train_general(self, epochs, n_features, batch_size):
        is_bidir = self.model_generator is not StackedLSTM
        zero_states = [np.zeros((batch_size, self.layer_sizes[0]))] * len(self.layer_sizes) * 2 * (2 if is_bidir else 1)
        y_train_list = [self.y_train[:, :, :i + 1] for i in range(self.y_train.shape[2])]
        y_val_list = [self.y_val[:, :, :i + 1] for i in range(self.y_val.shape[2])]
        history = self.gen_model.fit(self.X_train, y_train_list,
                                     validation_data=(self.X_val, y_val_list),
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

    def train_spec(self, epochs):
        # is_bidir = self.model_generator is not StackedLSTM
        # Create the context model, set the decoder = the gen model
        history = self.spec_model.fit([self.X_train] + self.stock_list, self.y_train,
                                      validation_data=([self.X_val] + self.stock_list, self.y_val),
                                      batch_size=self.batch_size, epochs=epochs, shuffle=False,
                                      )
        # write_to_csv(f'plot_data/spec/loss/{filename}.csv', history.history)
        # spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(self.X_train),
        #                                      layer_sizes=self.layer_sizes,
        #                                      return_states=True, decoder=self.spec_model.decoder, is_bidir=is_bidir)
        # spec_pred_model.set_weights(self.spec_model.get_weights())
        return history.history['loss'], history.history['val_loss']

    def generate_general_model_results(self, scaler_y, y_type, title, filename):
        model = self.gen_pred_model
        X = np.concatenate((self.X_train, self.X_val), axis=1)
        y = np.concatenate((self.y_train, self.y_val), axis=1)
        n_stocks = self.X_train.shape[0]
        result = model.predict([X])
        results_inverse_scaled = scaler_y.inverse_transform(result.reshape(n_stocks, -1))
        y_inverse_scaled = scaler_y.inverse_transform(y.reshape(n_stocks, -1))
        training_size = self.X_train.shape[1]

        result_train = results_inverse_scaled[:, :training_size].reshape(n_stocks, -1)
        result_val = results_inverse_scaled[:, training_size:].reshape(n_stocks, -1)

        y_train = y_inverse_scaled[:, :training_size].reshape(n_stocks, -1)
        y_val = y_inverse_scaled[:, training_size:].reshape(n_stocks, -1)

        val_evaluation = evaluate(result_val, y_val, y_type)
        train_evaluation = evaluate(result_train, y_train, y_type)
        print('Val: ', val_evaluation)
        print('Training:', train_evaluation)
        plot_one(f'{title}: Training', [result_train[0], y_train[0]], ['Predicted', 'True value'], ['Day', 'Change $'],
                 f'{filename}-train.png')
        plot_one(f'{title}: Test', [result_val[0], y_val[0]], ['Predicted', 'True value'], ['Day', 'Change $'],
                 f'{filename}-val.png')
        np.savetxt(f'{filename}-y.txt', y_inverse_scaled.reshape(-1))
        np.savetxt(f"{filename}-result.txt", results_inverse_scaled.reshape(-1))
        return {'training': train_evaluation, 'validation': val_evaluation}
