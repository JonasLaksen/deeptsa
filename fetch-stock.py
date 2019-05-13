from json import dump

from requests import get

api_key = '78e4f115c0960e419532bdbed431fc67'

stocks = ['']

for stock in stocks:
    test = get(f'http://api.stockfluence.com/fund/CSCO/history/2000-01-01/2019-03-05?apikey={api_key}').json()
    with open(f'data/{stock}.json', 'w') as file:
        dump(test, file)
