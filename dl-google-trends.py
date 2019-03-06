import pytrends
from pytrends.request import TrendReq
import pandas as pd
import calendar

if __name__ == '__main__':
    pytrends = TrendReq(hl='en-US', tz=60)

    stock = 'DIS'
    kw_list = [stock]


    # pytrends.build_payload(kw_list=kw_list, cat=0, timeframe='all', geo='', gprop='')
    #
    # results = pytrends.interest_over_time()
    #
    # results.to_csv('trends_data/' + stock + '_trends_all.csv')

    for year in range(2012, 2020):
        for month in range(1,13):
            day = calendar.monthrange(year, month)[1]
            str_month = ('0' + str(month) if month < 10 else str(month))
            str_day = ('0' + str(day) if day < 10 else str(day))
            timeframe = '' + str(year) + '-' + str_month + '-' + '01' + ' ' + \
                        str(year) + '-' + str_month + '-' + str_day
            pytrends.build_payload(kw_list=kw_list, cat=0, timeframe=timeframe, geo='', gprop='')

            results = pytrends.interest_over_time()
            results.to_csv('trends_data/' + stock + '_trends_' + str(year) + str_month + '.csv')


