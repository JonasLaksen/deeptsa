import numpy as np
import unittest

from src.utils import evaluate, load_data, get_features


class TestDataset(unittest.TestCase):

    def test_apple_stock (self):
        print('ok')
        X, y, y_dir, X_stocks, scaler_y = load_data(get_features(), y_type='next_price')
        X, y, X_stocks = X[0:1,].reshape(X.shape[1], -1),y[0:1,].reshape(-1), X_stocks[0:1,].reshape(-1)
        assert X_stocks[0] == 'AAPL'
        inverse_scaled_y = scaler_y.inverse_transform(y.reshape(1,-1))
        last_next_price = inverse_scaled_y[0,-1]
        assert last_next_price == 175.85
        next_last_next_price = y[-2]
        last_price = X[-1,0]
        assert next_last_next_price == last_price

if __name__ == '__main__':
    unittest.main()
