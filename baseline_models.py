import numpy as np

from utils import load_data

feature_list = ['positive', 'negative', 'neutral', 'open', 'high', 'low', 'volume', 'price']


def naive_model(y_test):




def main():
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    scaler_y = load_data(feature_list)

    X_train = np.append(X_train, X_val, axis=1)
    y_train = np.append(y_train, y_val, axis=1)



main()