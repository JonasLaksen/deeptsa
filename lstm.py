import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.losses import MAE
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from models.spec_network import SpecializedNetwork
from models.stacked_lstm import StackedLSTM
from utils import get_feature_list_lags, group_by_stock, plot


def main(gen_epochs=0, spec_epochs=0, load_gen=True, load_spec=False, model_generator=StackedLSTM, layer_sizes=[41],
         copy_weights_from_gen_to_spec=False, feature_list=[]):
    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.dropna()
    n_features = len(feature_list)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(data[feature_list].values)
    y = scaler_y.fit_transform(data['next_price'].values.reshape(-1, 1))
    X = np.append(data['stock'].values.reshape(-1, 1), X, axis=1)
    y = np.append(data['stock'].values.reshape(-1, 1), y, axis=1)

    X_train, X_val, X_test = group_by_stock(X)
    y_train, y_val, y_test = group_by_stock(y)

    batch_size = X_train.shape[0]
    is_bidir = model_generator is not StackedLSTM
    zero_states = [np.zeros((batch_size, layer_sizes[0]))] * len(layer_sizes) * 2 * (2 if is_bidir else 1)
    stock_list = [np.arange(len(X_train)).reshape((len(X_train), 1, 1))]

    gen_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                return_states=False)
    if load_gen:
        gen_model.load_weights('weights/gen.h5')
        print('Loaded generalised model')

    # Create the general model
    gen_model.compile(optimizer=Adam(0.001), loss=MAE)
    history = gen_model.fit([X_train] + zero_states, y_train, validation_data=([X_val] + zero_states, y_val), epochs=gen_epochs,
                  shuffle=False,
                  callbacks=[ModelCheckpoint('weights/gen.h5', period=10, save_weights_only=True), ])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    gen_pred_model = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size,
                                     return_states=True)
    gen_pred_model.set_weights(gen_model.get_weights())

    # Create the context model, set the decoder = the gen model
    decoder = model_generator(n_features=n_features, layer_sizes=layer_sizes, batch_size=batch_size, return_states=True)
    if copy_weights_from_gen_to_spec:
        decoder.set_weights(gen_model.get_weights())
    spec_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                    decoder=decoder, is_bidir=is_bidir)
    spec_model.compile(optimizer=Adam(0.001), loss=MAE)
    if load_spec:
        spec_model.load_weights('weights/spec.h5')
        print('Loaded specialised model')

    spec_model.fit([X_train] + stock_list, y_train, validation_data=([X_val] + stock_list, y_val),
                   batch_size=batch_size, epochs=spec_epochs, shuffle=False,
                   callbacks=[ModelCheckpoint('weights/spec.h5', period=1, save_weights_only=True)])
    spec_pred_model = SpecializedNetwork(n_features=n_features, num_stocks=len(y), layer_sizes=layer_sizes,
                                         return_states=True, decoder=spec_model.decoder, is_bidir=is_bidir)
    spec_pred_model.set_weights(spec_model.get_weights())

    # The place for plotting stuff
    # Only plot if the epoch > 0
    for model in ([gen_pred_model] if gen_epochs > 0 else []) + ([spec_pred_model] if spec_epochs > 0 else []):
        has_context = isinstance(model, SpecializedNetwork)
        # If general model, give zeros as input, if context give stock ids as input
        init_state = model.encoder.predict(stock_list) if has_context else zero_states

        if has_context:
            model = model.decoder

        result_train, *new_states = model.predict([X_train] + init_state)
        result_val, *new_states = model.predict([X_val] + new_states)
        result_test, *_ = model.predict([X_test] + new_states)

        # Plot only inverse transformed results for one stock
        stock_id = 10
        result_train, result_val, result_test, y_train_inv, y_val_inv, y_test_inv = map(
            lambda x: scaler_y.inverse_transform(x[stock_id]).reshape(-1),
            [result_train, result_val, result_test, y_train, y_val, y_test])

        [plot(*x) for x in [(f'training {"spec" if has_context else "gen"}', result_train, y_train_inv),
                            (f'validation {"spec" if has_context else "gen"}', result_val, y_val_inv)]]
    plt.show()


# feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']
feature_list = ['volume', 'price']
feature_list = get_feature_list_lags(feature_list, lags=2)
feature_list = feature_list + ['open', 'high', 'low']

main(
    copy_weights_from_gen_to_spec=False,
    feature_list=feature_list,
    gen_epochs=200,
    spec_epochs=0,
    layer_sizes=[41] * 1,
    load_gen=False,
    load_spec=False,
    # model_generator=bidir_lstm_seq.build_model,
    model_generator=StackedLSTM,
)
