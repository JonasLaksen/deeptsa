from preprocess import data_from_stock
import pandas as pd

dfs = [data_from_stock('GOOG'), data_from_stock('AAPL'), data_from_stock('DIS'), data_from_stock('TSLA'), data_from_stock('TWTR')]

pd.concat(dfs).to_csv('dataset.csv')



