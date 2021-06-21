import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import nltk
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
import preprocessor as p
import pickle
import warnings
warnings.filterwarnings("ignore")

clist =  pd.read_json('.\eng_contractions.txt', typ='series')
clist = clist.to_dict()

c_re = re.compile('(%s)' % '|'.join(clist.keys()))

def expandContractions(text, c_re=c_re):
    def replace(match):
        return clist[match.group(0)]
    return c_re.sub(replace, text)


BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = str(tweet)
        tweet = tweet.lower()
        tweet = BAD_SYMBOLS_RE.sub(' ', tweet)
        tweet = p.clean(tweet)
        
        #expand contraction
        tweet = expandContractions(tweet)

        #remove punctuation
        tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())

        #stop words
        stop_words = set(stopwords.words('english'))
        word_tokens = nltk.word_tokenize(tweet) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        tweet = ' '.join(filtered_sentence)
        
        cleaned_tweets.append(tweet)
        
    return cleaned_tweets


MAX_SEQ_LENGTH = 140
def token(x):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    X_test = tokenizer.texts_to_sequences(x)
    X_test = pd.DataFrame(tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LENGTH))
    return X_test

MAX_NB_WORDS = 10000
EMBEDDING_DIM = 300
def create_model():
        
        model = tf.keras.Sequential([
                              tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM,input_length = 140),
                              tf.keras.layers.LSTM(300),
                              tf.keras.layers.Dense(100, activation="relu"),
                              tf.keras.layers.Dropout(0.25),
                              tf.keras.layers.Dropout(0.2),
                              tf.keras.layers.Dense(1, activation='sigmoid')])

        return model


def model1(y):
    m = create_model()
    m.load_weights('./model.hdf5')
    output = (m.predict(y)>0.5).astype("int32")
    return output