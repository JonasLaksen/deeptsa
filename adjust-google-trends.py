import pandas as pd

if __name__ == '__main__':
    # stock_list = ['FB', 'AMZN', 'KO', 'PFE', 'WMT', 'HD', 'BIDU', 'QCOM', 'NFLX', 'NVDA', 'AMD', 'INTC']
    stock_list = ['NVDA', 'AMD', 'INTC']

    for stock in stock_list:
        trends_all = pd.read_csv('trends_data/' + stock + '_trends_all.csv')
        row_num = 0


        trends_all = trends_all[trends_all['date'] >= '2012-01-01'].reset_index()


        trends_adjusted = pd.DataFrame()
        for year in range(2012, 2020):
            for month in range(1, 13):
                if(year == 2019 and month >= 4):
                    continue

                str_month = ('0' + str(month) if month < 10 else str(month))

                trend_month = pd.read_csv('trends_data/' + stock + '_trends_' + str(year) + str_month + '.csv')

                trend_month[stock] = trend_month[stock]*(trends_all[stock][row_num]/100)

                print(stock, trends_all[stock][row_num])
                row_num += 1

                trends_adjusted = pd.concat([trends_adjusted, trend_month])

        trends_adjusted.to_csv('trends_data/' + stock + 'trends.csv')


