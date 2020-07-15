from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
import numpy as np


class Scaler():

    def __init__(self):
        self.scalers = []

    # (43,1660,2)
    def fit_on_training_and_transform_on_training_and_test(self,X_train, X_test, feature_range=(0,1)):
        scalers = []
        X_train_transformed = []
        X_test_transformed = []
        for i in range(X_train.shape[0]):
            scaler = MinMaxScaler(feature_range=feature_range)
            # scaler = FunctionTransformer(lambda x: x)
            X_train_transformed.append(scaler.fit_transform(X_train[i,:,]))
            X_test_transformed.append(scaler.transform(X_test[i,:,]))
            scalers.append(scaler)

        X_train_transformed_arr = np.asarray(X_train_transformed)
        X_test_transformed_arr = np.asarray(X_test_transformed)
        self.scalers = scalers
        return X_train_transformed_arr, X_test_transformed_arr

    def transform(self, X):
        X_transformed = []
        for i in range(X.shape[0]):
            scaler = self.scalers[i]
            X_transformed.append(scaler.transform(X[i,:,:]))
        return np.asarray(X_transformed)

    def inverse_transform(self, X):
        X_inverse_transformed = []
        for i in range(X.shape[0]):
            scaler = self.scalers[i]
            X_inverse_transformed.append(scaler.inverse_transform(X[i,:,]))
        return np.asarray(X_inverse_transformed)
