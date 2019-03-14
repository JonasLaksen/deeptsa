from preprocess import data_from_stock
import pandas as pd

dfs = [data_from_stock('GOOG'), data_from_stock('AAPL'), data_from_stock('DIS'), data_from_stock('TSLA'), data_from_stock('TWTR')]

def add_prev_feature(df, feature, n):
    for i in range(n):
        df[f'prev_{feature}_{i}'] = df[feature].shift(-(i+1))

for df in dfs:
    df['next_price'] = df['price'].shift(1)
    for feature in ['price','volume','trendscore','positive','negative','neutral']:
        add_prev_feature(df, feature, 2)

dfs = map(lambda x: x[1:-2], dfs)

#print(dfs[0][:5])
pd.concat(dfs).to_csv('dataset.csv')

