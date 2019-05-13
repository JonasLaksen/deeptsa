from json import dump

import pytrends
from pytrends.request import TrendReq
import pandas as pd
import calendar

from requests import get

api_key = '78e4f115c0960e419532bdbed431fc67'

if __name__ == '__main__':
    pytrends = TrendReq(hl='en-US', tz=60)
    stock_list = ['CSCO', 'CTSH', 'SBUX', 'EBAY', 'MDLZ', 'ATVI', 'EA', 'WDC', 'TXN', 'PEP', 'EXPE', 'ADBE',
                  'COST', 'HAS', 'SYMC', 'MSFT', 'SIRI', 'MU', 'GILD', 'MRVL', 'ODP', 'MAT', 'GRMN', 'PHG',
                  'BA', 'IBM', 'MCD', 'GS']
    stock = 'TWTR'


    # Download sentiment
    sentiment = get(f'http://api.stockfluence.com/fund/CSCO/history/2000-01-01/2019-03-05?apikey={api_key}').json()
    with open(f'data/{stock}.json', 'w') as file:
        dump(sentiment, file)
    kw_list = [stock + ' stock']


    pytrends.build_payload(kw_list=kw_list, cat=0, timeframe='all', geo='', gprop='')

    results = pytrends.interest_over_time()

    results.to_csv('trends_data/' + stock + '_trends_all.csv')

    for year in range(2012, 2020):
        for month in range(1,13):
            if year == 2019 and month == 4:
                break
            day = calendar.monthrange(year, month)[1]
            str_month = ('0' + str(month) if month < 10 else str(month))
            str_day = ('0' + str(day) if day < 10 else str(day))
            timeframe = '' + str(year) + '-' + str_month + '-' + '01' + ' ' + \
                        str(year) + '-' + str_month + '-' + str_day
            pytrends.build_payload(kw_list=kw_list, cat=0, timeframe=timeframe, geo='', gprop='')

            results = pytrends.interest_over_time()
            results.to_csv('trends_data/' + stock + '_trends_' + str(year) + str_month + '.csv')


