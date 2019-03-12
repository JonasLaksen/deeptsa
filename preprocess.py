import pandas as pd
import matplotlib.pyplot as plt

#Returns X, y
def data_from_stock(stock, show_plot=False):
    price_data = pd.read_csv(f'data/{stock} Historical Data.csv')[["Date", "Price", "Vol."]]
    price_data['price'] = (price_data['Price'].replace('[\$,)]', '', regex=True).astype(float))
    price_data["date"] = pd.to_datetime(price_data["Date"])
    price_data["volume"] = (price_data['Vol.'].replace('-','0', regex = True ).replace('K','e3', regex = True).replace('M', 'e6', regex = True).astype(float).fillna('0.00'))
    price_data = price_data.set_index("date")

    if show_plot:
        x,y=price_data.index.values, price_data.values
        plt.plot(x,y)
        plt.show()

    sentiment_data = pd.read_json(f'data/{stock}.json').transpose()[["positive","negative","neutral"]]
    sentiment_data.index.name = "date"

    trends_data = pd.read_csv(f'data/{stock}trends.csv')
    trends_data["date"] = pd.to_datetime(trends_data["date"])
    trends_data = trends_data.set_index("date")
    trends_data["trendscore"] = trends_data[stock]


    joined_data = price_data.join(sentiment_data, on="date")[["volume", "positive","negative", "neutral", "price"]]
    joined_data = joined_data.join(trends_data, on="date", how="inner")
    joined_data['stock'] = stock.upper()
    return joined_data[["stock", "volume", "positive", "negative", "neutral", "trendscore", "price"]]
