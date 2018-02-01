import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


def read_tweets(filepath):
    tweets = pd.read_csv(filepath, error_bad_lines=False)
    tweets = tweets[['Sentiment', 'SentimentText']]
    tweets.columns = ['sentiment','tweet']
    return tweets

def tweet_train_test_split(tweet_dataset, **kwargs):
    train, test = train_test_split(tweet_dataset, **kwargs)  
    
    X_train, y_train = train[['tweet']],train[['sentiment']]
    X_test, y_test = test[['tweet']],test[['sentiment']]
    return (X_train, y_train),(X_test, y_test)


class TweetPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_nr_words=20000, max_sequence_length=60):
        self.max_nr_words = max_nr_words
        self.max_sequence_length = max_sequence_length
        
        self.tokenizer = Tokenizer(num_words=max_nr_words)
        
    def fit(self, X, y=None):
        texts = X
        self.tokenizer.fit_on_texts(texts)   
        return self

    def transform(self, X, y=None, copy=None):
        texts = X
        X = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(X, maxlen=self.max_sequence_length, padding='post', truncating='post')
        X = np.stack(X, axis=0)
        if y is None:
            return X
        else:
            y = to_categorical(y, 2)
            return X, y  
    
    def fit_transform(self, X, y=None, copy=None):
        self.fit(X,y)        
        return self.transform(X, y)  
