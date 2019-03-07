import pandas as pd
import matplotlib.pyplot as plt

#Returns X, y
def data_from_stock(stock, show_plot=False):
    price_data = pd.read_csv(f'data/{stock} Historical Data.csv')[["Date", "Price"]]
    price_data['Price'] = (price_data['Price'].replace('[\$,)]', '', regex=True).astype(float))
    price_data["Date"] = pd.to_datetime(price_data["Date"])
    price_data = price_data.set_index("Date")

    if show_plot:
        x,y=price_data.index.values, price_data.values
        plt.plot(x,y)
        plt.show()

    sentiment_data = pd.read_json(f'data/{stock}.json').transpose()[["positive","negative","neutral"]]
    sentiment_data.index.name = "Date"

    trends_data = pd.read_csv(f'data/{stock}trends.csv')
    trends_data["Date"] = pd.to_datetime(trends_data["date"])
    trends_data = trends_data.set_index("Date")
    trends_data["trendscore"] = trends_data[stock]


    joined_data = price_data.join(sentiment_data, on="Date")[["positive","negative", "neutral", "Price"]]
    joined_data = joined_data.join(trends_data, on="Date", how="inner")
    joined_data['stock'] = stock.upper()
    joined_data['price'] = joined_data['Price']
    return joined_data[["stock","positive", "negative", "neutral", "trendscore", "price"]]
