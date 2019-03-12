from keras import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D


def create_model(x_train,  y_train):
    model = Sequential()
    model.add(Conv1D(64, 5, input_dim=x_train.shape[1], activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=150, batch_size=10)
    return model


def evaluate(model, x, y):
    scores = model.evaluate(x, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
