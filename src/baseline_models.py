import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR

from src.utils import load_data, evaluate

feature_list = ['price', 'high', 'low', 'open', 'volume', 'direction',
                'neutral_prop', 'positive_prop', 'negative_prop', 'negative', 'positive', 'neutral',
                'trendscore']


def naive_model(y_val, y_test, scaler_y):
    # result = np.append(y_val[:, -1:], y_test[:, :-1], axis=1)
    result = scaler_y.transform(np.zeros((y_val.shape[0], 167)))

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
    X, y, y_dir, _, scaler_y = load_data(feature_list, y_type='next_change')
    X, y, y_dir = X[0:1,:], y[0:1,:], y_dir[0:1,:]

    training_size = int(.9 * len(X[0]))
    X_train, y_train = X[:, :training_size], y[:, :training_size]
    X_test, y_test = X[:, training_size:], y[:, training_size:]

    result, y = naive_model(y_train, y_test, scaler_y)
    # result, y = linear_regression(X_train, X_test, y_train, y_test)
    # result, y = ridge_regression(X_train, X_test, y_train, y_test)

    # Not in use
    # result, y = gaussian_process(X_train, X_val, y_train, y_val)
    # result, y = svm(X_train, X_test, y_train, y_test)

    result = scaler_y.inverse_transform(result.reshape(X.shape[0], -1))
    y = scaler_y.inverse_transform(y.reshape(X.shape[0], -1))

    # plot("Baseline model", stock_list, result, y)

    print(evaluate(result, y, y_type='next_change'))

main()
#
# {'MAPE': 99.28900686324292, 'MAE': 2.327998885949032, '
# MSE': 53.22970058487674, 'DA': 0.5173374181868822}
