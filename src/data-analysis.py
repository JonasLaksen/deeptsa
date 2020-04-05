import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import constants


def plot_correlation(data, x_name, y_name):
    sns.jointplot(x=x_name, y=y_name, data=data)


if __name__ == '__main__':
    all_data = pd.read_csv('dataset_v2.csv')
    for stock in constants.stock_list:


        data = all_data[all_data['stock'] == stock]
        plot_correlation(data, 'direction', 'next_direction')

        plt.show()
        break

        # acf = plot_acf(data[['price']].values, lags=200, title=stock + ' Autocorrelation')
        # pacf = plot_pacf(data[['price']].values, lags=85, title=stock + ' Partial Autocorrelation')
        # acf.savefig('./figures/' + stock + '_acf.png')
        # plt.close(acf)
        # pacf.savefig('./figures/' + stock + '_pacf.png')
        # plt.close(pacf)