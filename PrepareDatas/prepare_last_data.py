import numpy as np
import pandas as pd


### FEATURE GENERATION ###

# MOVING AVERAGE
def setMovingAverage(data, symbol):
    data['moving_average'] = data[f'{symbol}_price'].rolling(window=7).mean()
    data['moving_average'] = data['moving_average'].interpolate(method='linear')
    data = data.dropna(subset=['moving_average'])

    return data


# RELATIVE STRENGTH INDEX (RSI)
def setRSI(data, symbol):
    window_size = 14

    # Calculate price changes
    data['price_change'] = data[f'{symbol}_price'].diff()
    data['price_change'] = data['price_change'].fillna(0)

    # Calculate gains and losses
    data['gain'] = data['price_change'].where(data['price_change'] > 0, 0)
    data['loss'] = data['price_change'].where(data['price_change'] < 0, 0)

    # Calculate average gains and average losses over the window size
    data['avg_gain'] = data['gain'].rolling(window=window_size).mean()
    data['avg_loss'] = data['loss'].rolling(window=window_size).mean()

    # Replace null and infinity values with 0
    data['avg_gain'] = data['avg_gain'].replace([np.inf, -np.inf, np.nan], 0)
    data['avg_loss'] = data['avg_loss'].replace([np.inf, -np.inf, np.nan], 0)

    # Calculate relative strength (RS)
    data['rs'] = data['avg_gain'] / data['avg_loss']

    # Replace infinity values in RS with 0
    data['rs'] = data['rs'].replace([np.inf, -np.inf], 0)
    data['rs'] = data['rs'].fillna(0)

    # Calculate RSI
    data['rsi'] = 100 - (100 / (1 + data['rs']))

    # Replace infinity values in RSI with 0
    data['rsi'] = data['rsi'].replace([np.inf, -np.inf], 0)
    data['rsi'] = data['rsi'].fillna(0)

    return data


# MOVING AVERAGE CONVERGENCE DIVERGENCE (MACD)

def setMACD(data, symbol):
    # Assuming you have a DataFrame called 'data' with f'{symbol}_price' column
    # Calculate the 12-day EMA
    ema12 = data[f'{symbol}_price'].ewm(span=12, adjust=False).mean()

    # Calculate the 26-day EMA
    ema26 = data[f'{symbol}_price'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    macd_line = ema12 - ema26

    # Calculate the 9-day EMA of the MACD line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line

    # Add the calculated columns to the DataFrame
    data['macd_line'] = macd_line
    data['signal_line'] = signal_line
    data['macd_histogram'] = macd_histogram

    # Forward fill the null values in the MACD-related columns
    data['macd_line'].ffill(inplace=True)
    data['signal_line'].ffill(inplace=True)
    data['macd_histogram'].ffill(inplace=True)

    return data


def setMovement(data, symbol):
    data['Price_Diff'] = data[f'{symbol}_price'].shift(-1) - data[f'{symbol}_price']
    data['movement'] = np.where(data['Price_Diff'] > 0, 1, 0)
    data['Price_Diff'] = data['Price_Diff'].fillna(0)
    data = data.dropna()
    data = data.drop('Price_Diff', axis=1)
    data = data.drop('price_change', axis=1)

    return data


def floatConversion(data):
    gl_float = data.select_dtypes(include=['float'])
    converted_float = gl_float.apply(pd.to_numeric, downcast='float')
    compare_floats = pd.concat([gl_float.dtypes, converted_float.dtypes], axis=1)
    compare_floats.columns = ['Before', 'After']
    compare_floats.apply(pd.Series.value_counts)
    data[converted_float.columns] = converted_float

    return data


def generateData(data, symbol):
    data = setMovingAverage(data, symbol)
    data = setRSI(data, symbol)
    data = setMACD(data, symbol)
    data = setMovement(data, symbol)
    data = floatConversion(data)

    return data
