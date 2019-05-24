import pandas as pd
import matplotlib.pyplot as plt
from constants import stock_list

#Returns X, y
def data_from_stock(stock, show_plot=False):
    price_data = pd.read_csv(f'data/{stock} Historical Data.csv')[["Date", "Price", "Vol.", "Open", "High", "Low"]]
    price_data['price'] = (price_data['Price'].replace('[\$,)]', '', regex=True).astype(float))
    price_data['open'] = (price_data['Open'].replace('[\$,)]', '', regex=True).astype(float))
    price_data['high'] = (price_data['High'].replace('[\$,)]', '', regex=True).astype(float))
    price_data['low'] = (price_data['Low'].replace('[\$,)]', '', regex=True).astype(float))
    price_data["date"] = pd.to_datetime(price_data["Date"])
    price_data["volume"] = (price_data['Vol.'].replace('-','0', regex = True ).replace('K','e3', regex = True).replace('M', 'e6', regex = True).astype(float).fillna('0.00'))
    price_data = price_data.set_index("date")

    if show_plot:
        x,y=price_data.index.values, price_data.values
        plt.plot(x,y)
        plt.show()

    sentiment_data = pd.read_json(f'data/{stock}.json').transpose()[["positive","negative","neutral"]]
    sentiment_data.index.name = "date"
    sentiment_data["positive"] = sentiment_data["positive"].astype(float)
    sentiment_data["negative"] = sentiment_data["negative"].astype(float)
    sentiment_data["neutral"] = sentiment_data["neutral"].astype(float)

    trends_data = pd.read_csv(f'data/{stock}trends.csv')
    trends_data["date"] = pd.to_datetime(trends_data["date"])
    trends_data = trends_data.set_index("date")
    trends_data["trendscore"] = trends_data[stock + ' stock'].astype(float)/100


    joined_data = price_data.join(sentiment_data, on="date")[["volume", "positive","negative", "neutral", "price",
                                                              "open", "high", "low"]]
    joined_data = joined_data.join(trends_data, on="date", how="inner")
    joined_data['stock'] = stock.upper()
    joined_data = joined_data[["stock", "volume", "positive", "negative", "neutral", "trendscore", "price",
                        "open", "high", "low"]]
    return joined_data.reindex(index=joined_data.index[::-1])


def add_prev_feature(df, feature, n):
    for i in range(n):
        df[f'prev_{feature}_{i}'] = df[feature].shift(-(i+1))

def write_to_dataset_file():
    dfs = list(map(lambda x: data_from_stock(x), stock_list))
    for df in dfs:
        df['next_price'] = df['price'].shift(1)
        for feature in ['price','volume','trendscore','positive','negative','neutral']:
            add_prev_feature(df, feature, 2)

        # df['change'] = df['next_price'] - df['price']
        # df['change_percent'] = (df['next_price'] - df['price'])/df['price']
        total_sent = df['positive'] + df['negative'] + df['neutral']
        df['positive_prop'] = df['positive'] / total_sent
        df['negative_prop'] = df['negative'] / total_sent
        df['neutral_prop'] = df['neutral'] / total_sent

    dfs = map(lambda x: x[1:-2], dfs)
    pd.concat(dfs).to_csv('dataset_v2.csv')

if __name__ == '__main__':
    write_to_dataset_file()

