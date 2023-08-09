import warnings
import numpy as np
import pandas as pd
import nltk  # natural language processing
from nltk.corpus import stopwords
from nltk.stem.porter import *
from tqdm import tqdm
import h5py
import tensorflow as tf
from transformers import TFRobertaModel
from transformers import RobertaTokenizerFast

warnings.filterwarnings("ignore")
nltk.download("stopwords")

symbol = symbol

modelPath = f'Models/{symbol}/Roberta Model'
dataFilePath = f'Data Files/{symbol}'


### SENTIMENT DATA ###

# Define the custom objects
custom_objects = {
    'TFRobertaModel': TFRobertaModel.from_pretrained('roberta-base')
}

# Load the HDF5 model
with h5py.File(f'{modelPath}/roberta-model.h5', 'r') as file:
    # Load the model
    model = tf.keras.models.load_model(file, custom_objects=custom_objects, compile=False)

sentiment_df = pd.read_csv(f'{dataFilePath}/csv Files/bitcoin.com_news_url.csv', encoding='utf-16')
# Convert the datetime column to datetime type
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])


def tweet_to_words(tweet):
    ''' Convert tweet text into a sequence of words '''
    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words


cleantext = []

for item in tqdm(sentiment_df['title']):
    words = tweet_to_words(item)
    cleantext += [words]

sentiment_df['cleantext'] = cleantext


# Define the unlist function
def unlist(lst):
    words = ''
    for item in lst:
        words += item + ' '
    return words


# Apply the unlist function to all rows in the DataFrame
sentiment_df['cleantext'] = sentiment_df['cleantext'].apply(unlist)

X = sentiment_df['cleantext'].values

tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")

token_lens = []

for txt in X:
    tokens = tokenizer_roberta.encode(txt, max_length=250, truncation=True)
    token_lens.append(len(tokens))

max_length = np.max(token_lens)

MAX_LEN = 128


def tokenize_roberta(data, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer_roberta.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)


input_ids, attention_masks = tokenize_roberta(X, MAX_LEN)

result_roberta = model.predict([input_ids, attention_masks])

y_pred_roberta = np.zeros_like(result_roberta)
y_pred_roberta[np.arange(len(y_pred_roberta)), result_roberta.argmax(1)] = 1

predicted_class = result_roberta.argmax(1)

sentiment_df['predicted_class'] = predicted_class

# Pivot the DataFrame and calculate the count for each combination of 'date' and 'predicted_class'
result = sentiment_df.pivot_table(index='date', columns='predicted_class', aggfunc='size', fill_value=0)

print("Result:\n", result)

result.to_excel(f"{dataFilePath}/xlsx Files/bitcoin.com_news_Test_Data_Sentiments.xlsx")
result.to_csv(f"{dataFilePath}/csv Files/bitcoin.com_news_Test_Data_Sentiments.csv")

sentiment_df = pd.read_csv(f"{dataFilePath}/csv Files/bitcoin.com_news_Test_Data_Sentiments.csv")
# Convert the datetime column to datetime type
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# Sort the dataframe by the datetime column in ascending order
sentiment_df = sentiment_df.sort_values('date')

# Format the datetime column as "yyyy-mm-dd"
sentiment_df['date'] = sentiment_df['date'].dt.strftime('%Y-%m-%d')
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

column_mapping = {
    '0': 'negative',
    '1': 'neutral',
    '2': 'positive',
}

sentiment_df = sentiment_df.rename(columns=column_mapping)
sentiment_df = sentiment_df.reset_index(drop=True)

print("Sentiment_df :\n", sentiment_df)

sentiment_df.info()

### BITCOIN DATA ###

df = pd.read_csv(f'{dataFilePath}/csv Files/prices.csv')

# Convert the datetime column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Sort the dataframe by the datetime column in ascending order
df = df.sort_values('date')

# Format the datetime column as "yyyy-mm-dd"
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

# df['price'] = df['price'].str.replace(',', '')
df['price'] = pd.to_numeric(df['price'])

column_mapping = {
    'date': 'date',
    'price': f'{symbol}_price',
}

df = df.rename(columns=column_mapping)
df['date'] = pd.to_datetime(df['date'])
df = df.reset_index(drop=True)

print("df.info() : ", df.info())

### CURRENCY DATA ###

currency_df = pd.read_csv('Data Files/currency/csv Files/currency_prices.csv')

column_mapping = {
    'date': 'date',
    'price': 'currency_price'
}

currency_df = currency_df.rename(columns=column_mapping)

# Convert the datetime column to datetime type
currency_df['date'] = pd.to_datetime(currency_df['date'])

# Sort the dataframe by the datetime column in ascending order
currency_df = currency_df.sort_values('date')

currency_df['currency_price'] = currency_df['currency_price'].replace(',', '.')
currency_df['currency_price'] = pd.to_numeric(currency_df['currency_price'])

currency_df = currency_df.reset_index(drop=True)

print("currency_df.info() : ", currency_df.info())

### FED INTEREST RATE ###

interest_df = pd.read_csv('Data Files/DFF Files/DFF.csv')

column_mapping = {
    'DATE': 'date',
    'DFF': 'interest_rate'}

interest_df = interest_df.rename(columns=column_mapping)

interest_df['date'] = pd.to_datetime(interest_df['date'])

print("interest_df.info() : ", interest_df.info())

### FINAL DATA ###

# Join the first two DataFrames based on the "date" column
merged_df = pd.merge(df, sentiment_df, on='date', how='left')
merged_df = pd.merge(merged_df, currency_df, on='date', how='left')
merged_df = pd.merge(merged_df, interest_df, on='date', how='left')

# final_df = pd.merge(merged_df, interest_df, on='date', how='left')
final_df = merged_df

# Fill null values with zero in the specified columns
final_df['negative'] = final_df['negative'].fillna(0)
final_df['neutral'] = final_df['neutral'].fillna(0)
final_df['positive'] = final_df['positive'].fillna(0)

final_df.isnull().sum()

final_df['currency_price'] = final_df['currency_price'].interpolate(method='linear')

final_df.isnull().sum()

final_df.to_excel(f"{dataFilePath}/xlsx Files/final_data_TEST_DATA.xlsx", index=False)
final_df.to_csv(f"{dataFilePath}/csv Files/final_data_TEST_DATA.csv", index=False)

### FEATURE GENERATION ###

# Importing of Data
data = pd.read_csv(f"{dataFilePath}/csv Files/final_data_TEST_DATA.csv")
data.set_index('date', inplace=True)

print(f"Total records:{data.shape}\n")
print(f"Data types of data columns: \n{data.dtypes}")

### MOVING AVERAGE ###

# Calculate the 7-day moving average
data['moving_average'] = data[f'{symbol}_price'].rolling(window=7).mean()
print("Moving average mean:\n", data)

data['moving_average'] = data['moving_average'].interpolate(method='linear')
data = data.dropna(subset=['moving_average'])
print("Moving average:\n", data)

### RELATIVE STRENGTH INDEX (RSI) ###

# Assuming you have a DataFrame called 'data' with a 'btc_price' column

# Define the window size for RSI calculation
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

# Display the updated DataFrame
print("RSI :\n", data)

### MOVING AVERAGE CONVERGENCE DIVERGENCE (MACD) ###

# Assuming you have a DataFrame called 'data' with a '{symbol}_price' column

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

data['Price_Diff'] = data[f'{symbol}_price'].shift(-1) - data[f'{symbol}_price']
data['movement'] = np.where(data['Price_Diff'] > 0, 1, 0)
data = data.dropna()
data = data.drop('Price_Diff', axis=1)
data = data.drop('price_change', axis=1)

print("MACD :\n", data)

data.to_excel(f"{dataFilePath}/xlsx Files/final_test_data.xlsx", index=True)
data.to_csv(f"{dataFilePath}/csv Files/final_test_data.csv", index=True)
