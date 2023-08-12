import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import emoji
import nltk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from tensorflow.keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

# set seed for reproducibility
seed = 42

# set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)

showPlot = showPlot
symbol = symbol

modelPath = f'Models/{symbol}/Roberta Model'


def conf_matrix(y, y_pred, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ['Negative', 'Neutral', 'Positive']
    ax = sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size": 25})
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17)
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    if showPlot: plt.show()


df = pd.read_excel(f'Data Files/{symbol}/xlsx Files/bitcoin.com_news_url.xlsx')

print(df.info())

# text length
df['text_length'] = df['title'].apply(len)
df[['text_length', 'title']].head()

df['text_length'].describe()

df['text_length'].hist(bins=50)

# g = sns.FacetGrid(df,col='label')
# g.map(plt.hist,'text_length')

# word cloud
nltk.download('stopwords')


def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    return s


df['clean_title'] = df['title'].apply(clean_text)

text = df['clean_title'].to_string().lower()
wordcloud = WordCloud(
    collocations=False,
    relative_scaling=0.5,
    stopwords=set(stopwords.words('english'))).generate(text)

plt.figure(figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig(f'Plot Files/{symbol}/wordcloud.png')
if showPlot: plt.show()


# CUSTOM DEFINED FUNCTIONS TO CLEAN THE TWEETS
# Clean emojis from text
def strip_emoji(text):
    return re.sub(emoji.get_emoji_regexp(), r"", text)  # remove emoji


# Remove punctuations, links, mentions and \r\n new line characters
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()  # remove \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list = string.punctuation + 'Ã' + '±' + 'ã' + '¼' + 'â' + '»' + '§'
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    return text


# clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in
                         re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))  # remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in
                          re.split('#|_', new_tweet))  # remove hashtags symbol from words in the middle of the sentence
    return new_tweet2


# Filter special characters such as & and $ present in some words
def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)


def remove_mult_spaces(text):  # remove multiple spaces
    return re.sub("\s\s+", " ", text)


texts_new = []
for t in df.title:
    texts_new.append(remove_mult_spaces(filter_chars(clean_hashtags(strip_all_entities(t)))))

df['text_clean'] = texts_new

print(df.describe())

plt.figure(figsize=(7, 5))
# ax = sns.countplot(x='text_length', data=df[df['text_length']<20], palette='mako')
# plt.title('Training comments with less than 20 words')
plt.yticks([])
# ax.bar_label(ax.containers[0])
plt.xlabel('word_count')
plt.ylabel('count')
plt.savefig(f'Plot Files/{symbol}/count_wordCount.png')
if showPlot: plt.show()

print(f" DF SHAPE: {df.shape}")

df = df[df['text_length'] > 5]

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

token_lens = []

for txt in df['text_clean'].values:
    tokens = tokenizer.encode(txt, max_length=512, truncation=True)
    token_lens.append(len(tokens))

max_len = np.max(token_lens)

df['token_lens'] = token_lens

df = df.sort_values(by='token_lens', ascending=False)
print(df.head(5))

df = df.sample(frac=1).reset_index(drop=True)

print(df.describe())


def unlist(list):
    words = ''
    for item in list:
        words += item + ' '
    return words


def compute_vader_scores(df, label):
    sid = SentimentIntensityAnalyzer()
    df["vader_neg"] = df[label].apply(lambda x: sid.polarity_scores(x)["neg"])
    df["vader_neu"] = df[label].apply(lambda x: sid.polarity_scores(x)["neu"])
    df["vader_pos"] = df[label].apply(lambda x: sid.polarity_scores(x)["pos"])
    df["vader_comp"] = df[label].apply(lambda x: sid.polarity_scores(x)["compound"])
    return df


df2 = compute_vader_scores(df, 'text_clean')

class0 = []
for i in range(len(df2)):
    if df2.loc[i, 'vader_neg'] > 0:
        class0 += [0]
    elif df2.loc[i, 'vader_pos'] > 0:
        class0 += [2]
    else:
        class0 += [1]

df['class'] = class0
df['class'].value_counts()

X = df['text_clean'].values
y = df['class'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

print("##################### Length #####################")
print(f'Total # of sample in whole dataset: {len(X_train) + len(X_test) + len(X_valid)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in validation dataset: {len(X_valid)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

print("##################### Shape #####################")
print(f'Shape of train dataset: {X_train.shape}')
print(f'Shape of validation dataset: {X_valid.shape}')
print(f'Shape of test dataset: {X_test.shape}')

print("##################### Percantage #####################")
print(f'Percentage of train dataset: {round((len(X_train) / (len(X_train) + len(X_test))) * 100, 2)}%')
print(f'Percentage of validation dataset: {round((len(X_test) / (len(X_train) + len(X_test))) * 100, 2)}%')

y_train_le = y_train.copy()
y_valid_le = y_valid.copy()
y_test_le = y_test.copy()

ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()
y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1)).toarray()

tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")

token_lens = []

for txt in X_train:
    tokens = tokenizer_roberta.encode(txt, max_length=512, truncation=True)
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


train_input_ids, train_attention_masks = tokenize_roberta(X_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize_roberta(X_valid, MAX_LEN)
test_input_ids, test_attention_masks = tokenize_roberta(X_test, MAX_LEN)


def create_model(bert_model, max_len=MAX_LEN):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')
    output = bert_model([input_ids, attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(output)
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(opt, loss=loss, metrics=accuracy)
    return model


roberta_model = TFRobertaModel.from_pretrained('roberta-base')

model = create_model(roberta_model, MAX_LEN)
model.summary()

sequence_length = 128  # the maximum length of sequences in your dataset

train_input_ids = sequence.pad_sequences(train_input_ids, maxlen=sequence_length, dtype='int32', value=0,
                                         padding='post', truncating='post')
train_attention_masks = sequence.pad_sequences(train_attention_masks, maxlen=sequence_length, dtype='int32', value=0,
                                               padding='post', truncating='post')

val_input_ids = sequence.pad_sequences(val_input_ids, maxlen=sequence_length, dtype='int32', value=0, padding='post',
                                       truncating='post')
val_attention_masks = sequence.pad_sequences(val_attention_masks, maxlen=sequence_length, dtype='int32', value=0,
                                             padding='post', truncating='post')

y_train = np.asarray(y_train)
y_valid = np.asarray(y_valid)

print("##################### Length #####################")
print(f'Total # of train_input_ids: {len(train_input_ids)}')
print(f'Total # of train_attention_masks: {len(train_attention_masks)}')

print(f'Total # of val_input_ids: {len(val_input_ids)}')
print(f'Total # of val_attention_masks: {len(val_attention_masks)}')

print(f'Total # of y_train_tf: {len(y_train)}')
print(f'Total # of y_valid_tf: {len(y_valid)}')

print("##################### Shape #####################")
print(f'Shape of train_input_ids: {train_input_ids.shape}')
print(f'Shape of train_attention_masks: {train_attention_masks.shape}')
print(f'Shape of val_input_ids: {val_input_ids.shape}')
print(f'Shape of val_attention_masks: {val_attention_masks.shape}')
print(f'Shape of y_train_tf: {y_train.shape}')
print(f'Shape of y_valid_tf: {y_valid.shape}')

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train model with early stopping
history_2 = model.fit(
    [train_input_ids, train_attention_masks], y_train,
    validation_data=([val_input_ids, val_attention_masks], y_valid),
    epochs=1,
    batch_size=128,
    callbacks=[early_stop]
)


def plot_training_hist(history):
    '''Function to plot history for accuracy and loss'''

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # first plot
    ax[0].plot(history.history['categorical_accuracy'])
    ax[0].plot(history.history['val_categorical_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'validation'], loc='best')

    # second plot
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(['train', 'validation'], loc='best')
    plt.savefig(f'Plot Files/{symbol}/plot_training_hist.png')
    if showPlot: plt.show()


plot_training_hist(history_2)

test_input_ids = pad_sequences(test_input_ids, maxlen=sequence_length, dtype='int32', value=0, padding='post',
                               truncating='post')
test_attention_masks = pad_sequences(test_attention_masks, maxlen=sequence_length, dtype='int32', value=0,
                                     padding='post', truncating='post')

result_roberta = model.predict([test_input_ids, test_attention_masks])

y_pred_roberta = np.zeros_like(result_roberta)
y_pred_roberta[np.arange(len(y_pred_roberta)), result_roberta.argmax(1)] = 1

conf_matrix = confusion_matrix(np.argmax(y_pred_roberta, axis=1), y_test_le)
print(f'Confussion Matrix: \n{conf_matrix}\n')

tn = conf_matrix[0, 0]
fp = conf_matrix[0, 1]
tp = conf_matrix[1, 1]
fn = conf_matrix[1, 0]

total = tn + fp + tp + fn
real_positive = tp + fn
real_negative = tn + fp

accuracy = (tp + tn) / total  # Accuracy Rate
precision = tp / (tp + fp)  # Positive Predictive Value
recall = tp / (tp + fn)  # True Positive Rate
f1score = 2 * precision * recall / (precision + recall)
specificity = tn / (tn + fp)  # True Negative Rate
error_rate = (fp + fn) / total  # Missclassification Rate
prevalence = real_positive / total
miss_rate = fn / real_positive  # False Negative Rate
fall_out = fp / real_negative  # False Positive Rate

print('Evaluation Metrics:')
print(f'Accuracy    : {accuracy}')
print(f'Precision   : {precision}')
print(f'Recall      : {recall}')
print(f'F1 score    : {f1score}')
print(f'Specificity : {specificity}')
print(f'Error Rate  : {error_rate}')
print(f'Prevalence  : {prevalence}')
print(f'Miss Rate   : {miss_rate}')
print(f'Fall Out    : {fall_out}')

print("")
print(f'Classification Report: \n{classification_report(np.argmax(y_pred_roberta, axis=1), y_test_le)}\n')
print("")

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

# Save the trained model and tokenizer
model.save(f'{modelPath}/roberta-model.h5')
tokenizer.save_pretrained(f'{modelPath}/tokenizer')
