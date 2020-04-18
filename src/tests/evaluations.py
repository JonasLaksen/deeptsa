import numpy as np
import unittest

from src.utils import evaluate


class TestEvaluations(unittest.TestCase):
    def test_da_direction(self):
        results = np.asarray([[[1], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ -2 ], [ -1 ]]])
        y = np.asarray([[[ 1 ], [ 1 ], [ 1 ], [ -1 ], [ 1 ], [ 1 ], [ 1 ], [ -1 ], [ -2 ]]])
        DA = evaluate(results, y, y_type='direction')['DA']
        assert DA.round(3) == .889

        r2 = np.asarray([1,1,1]).reshape((1,1,3))
        y2 = np.asarray([1,1,1]).reshape((1,1,3))
        DA2 = evaluate(r2, y2, y_type='direction')['DA']
        assert DA2.round(3) == 1

        r3 = np.asarray([1,1,-1]).reshape((1,1,3))
        y3 = np.asarray([1,1,1]).reshape((1,1,3))
        DA3 = evaluate(r3, y3, y_type='direction')['DA']
        assert DA3.round(3) == 0.667

        r4 = np.asarray([1,1,0]).reshape((1,1,3))
        y4 = np.asarray([1,1,1]).reshape((1,1,3))
        DA4 = evaluate(r4, y4, y_type='direction')['DA']
        assert DA4.round(3) == 0.667

    def test_da_price(self):
        results = np.asarray([[[1], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ 1 ], [ -2 ], [ -1 ]]])
        y = np.asarray([[[ 1 ], [ 1 ], [ 1 ], [ -1 ], [ 1 ], [ 1 ], [ 1 ], [ -1 ], [ -2 ]]])
        DA = evaluate(results, y, y_type='price')['DA']
        print(DA.round(3))
        assert DA.round(3) == .750

if __name__ == '__main__':
    unittest.main()