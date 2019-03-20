from preprocess import data_from_stock
import pandas as pd

stock_list = ['AAPL', 'AMD', 'AMZN', 'BIDU', 'DIS', 'FB', 'GOOG', 'HD', 'INTC', 'KO', 'NFLX', 'NVDA', 'PFE', 'QCOM',
              'TSLA', 'TWTR', 'WMT']

dfs = list(map(lambda x: data_from_stock(x), stock_list))

pd.concat(dfs).to_csv('dataset_v3.csv')



