import numpy as np

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from utils import load_data, plot, evaluate
from constants import stock_list

feature_list = ['price', 'high', 'low', 'open', 'volume', 'direction',
                'neutral_prop', 'positive_prop', 'negative_prop', 'negative', 'positive', 'neutral',
                'trendscore']


def naive_model(y_val, y_test):
    result = np.append(y_val[:,-1:], y_test[:,:-1], axis=1)

    return result, y_test


def svm(X_train, X_test, y_train, y_test):

    result = []

    for i in range(X_train.shape[0]):
        svm_model = SVR()
        svm_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = svm_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def linear_regression(X_train, X_test, y_train, y_test):

    result = []

    for i in range(X_train.shape[0]):
        linear_model = LinearRegression()
        linear_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = linear_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test

def ridge_regression(X_train, X_test, y_train, y_test):

    result = []

    for i in range(X_train.shape[0]):
        logistic_model = Ridge()
        y = y_train[i].reshape(-1)
        logistic_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = logistic_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test

def gaussian_process(X_train, X_test, y_train, y_test):
    result = []

    for i in range(X_train.shape[0]):
        gaussian_model = GaussianProcessRegressor()
        y = y_train[i].reshape(-1)
        gaussian_model.fit(X_train[i], y_train[i].reshape(-1))
        partial_result = gaussian_model.predict(X_test[i])

        result.append(partial_result)

    result = np.array(result)

    return result, y_test


def main():
    (X_train, X_val, X_test), \
    (y_train, y_val, y_test), \
    (y_train_dir, y_val_dir, y_test_dir), \
    scaler_y = load_data(feature_list)

    X_train = np.append(X_train, X_val, axis=1)
    y_train = np.append(y_train, y_val, axis=1)

    result, y = naive_model(y_val, y_test)
    # result, y = linear_regression(X_train, X_test, y_train, y_test)
    # result, y = ridge_regression(X_train, X_test, y_train, y_test)

    #Not in use
    # result, y = gaussian_process(X_train, X_val, y_train, y_val)
    # result, y = svm(X_train, X_test, y_train, y_test)

    result = scaler_y.inverse_transform(result.reshape(43,-1))
    y = scaler_y.inverse_transform(y.reshape(43,-1))

    # plot("Baseline model", stock_list, result, y)

    print(evaluate(result, y))

main()