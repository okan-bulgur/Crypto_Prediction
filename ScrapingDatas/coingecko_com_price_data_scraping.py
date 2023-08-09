import pandas as pd
import requests
from datetime import datetime


def get_historical_data(symbol, start_date, end_date):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range'
    params = {
        'vs_currency': 'usd',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp())
    }

    response = requests.get(url, params=params)
    data = response.json()
    print(data)

    if 'prices' in data:
        historical_data = data['Prices']
        return historical_data
    else:
        return None


def get_daily_closing_prices(symbol, start_date, end_date):
    url = f'https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range'
    params = {
        'vs_currency': 'usd',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp())
    }

    response = requests.get(url, params=params)
    prices_data = response.json()
    prices = pd.DataFrame(prices_data['prices'], columns=['time', 'prices'])
    prices['time'] = pd.to_datetime(prices['time'], unit='ms').dt.date
    prices.set_index('time', inplace=True)

    if 'prices' in prices_data:
        historical_data = prices_data['prices']
    else:
        return

    daily_prices = {}
    daily_prices_list = []

    for data_point in historical_data:
        new_price = {}
        timestamp = data_point[0] / 1000
        date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
        close_price = data_point[1]
        daily_prices[date] = close_price

    for date, close_price in daily_prices.items():
        new_price = {}
        new_price['date'] = date
        new_price['price'] = close_price
        daily_prices_list.append(new_price)

    prices_df = pd.DataFrame.from_records(daily_prices_list, columns=['date', 'price'])
    return prices_df


def createPriceFile(data):
    start_date_str = data.getStartDate()
    end_date_str = data.getEndDate()
    symbol = data.getCryptoType()

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    daily_closing_prices_crypto_df = get_daily_closing_prices(symbol, start_date, end_date)
    daily_closing_prices_usdt_df = get_daily_closing_prices('tether', start_date, end_date)
    daily_closing_prices_eurt_df = get_daily_closing_prices('tether-eurt', start_date, end_date)

    daily_closing_prices_currency_df = pd.merge(daily_closing_prices_usdt_df, daily_closing_prices_eurt_df, on='date',
                                                suffixes=('_df1', '_df2'))
    daily_closing_prices_currency_df['price'] = daily_closing_prices_currency_df['price_df1'] / \
                                                daily_closing_prices_currency_df['price_df2']
    daily_closing_prices_currency_df = daily_closing_prices_currency_df.drop({'price_df1', 'price_df2'}, axis=1)

    daily_closing_prices_crypto_df.to_csv(f'Data Files/{symbol}/csv Files/prices.csv', index=False)
    daily_closing_prices_crypto_df.to_excel(f'Data Files/{symbol}/xlsx Files/prices.xlsx', index=False)

    daily_closing_prices_currency_df.to_csv("Data Files/currency/csv Files/currency_prices.csv", index=False)
    daily_closing_prices_currency_df.to_excel("Data Files/currency/xlsx Files/currency_prices.xlsx", index=False)
