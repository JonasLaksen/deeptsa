import matplotlib.pyplot as plt
import pandas as pd

def main():
    stock = 'AMD'
    month_all = pd.read_csv('./trends_data/'+ stock + '_trends_all.csv')
    all_data = pd.read_csv('./trends_data/' + stock + 'trends.csv')

    month_all = month_all[month_all['date'] >= '2012-01-01']

    month_all = month_all.set_index(month_all['date'])

    all_data = all_data[['date', stock + ' stock']].set_index(all_data['date'])


    month_all.plot(legend=False)
    all_data.plot(legend=False)


    plt.show()



main()