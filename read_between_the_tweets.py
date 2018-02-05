import os

from sklearn.externals import joblib

from twitter_sentiment.preprocessing import read_tweets, tweet_train_test_split, TweetPreprocessor
from twitter_sentiment.architectures import arch_lstm, arch_conv1d, arch_attention, arch_attention36
from twitter_sentiment.model import TweetClassifier


MAX_WORDS = 20000
MAX_SEQ_LEN = 30
EMBEDDING_DIM = 50
ARCHITECTURE = arch_attention36
LOCAL_DIR = '/mnt/ml-team/minerva/unblackboxing_webinar'
EMBEDDING_MODEL_FILENAME = '/mnt/ml-team/minerva/pretrained/glove.twitter.27B.50d.txt'
PREP_DUMP_FILENAME = 'tweet_preprocessor.pkl'
CLASS_DUMP_FILENAME = 'tweetnetAttention36.h5'
DATA_FILEPATH ='/mnt/ml-team/minerva/unblackboxing_webinar/Sentiment Analysis Dataset.csv'
EMBEDDING_MODEL_FILEPATH = os.path.join(LOCAL_DIR, EMBEDDING_MODEL_FILENAME)
PREP_DUMP_FILEPATH = os.path.join(LOCAL_DIR, PREP_DUMP_FILENAME)
CLASS_DUMP_FILEPATH = os.path.join(LOCAL_DIR, CLASS_DUMP_FILENAME)

if __name__ == '__main__':
    
    tweet_dataset = read_tweets(DATA_FILEPATH)

    (X_train, y_train), (X_test,y_test) = tweet_train_test_split(tweet_dataset, train_size=0.8,
                                                                random_state=1234)
    
    tweet_prep = TweetPreprocessor(max_nr_words=MAX_WORDS, max_sequence_length=MAX_SEQ_LEN)
    X_train, y_train = tweet_prep.fit_transform(X=X_train['tweet'].values, y=y_train)
    X_test, y_test = tweet_prep.transform(X=X_test['tweet'].values, y=y_test)
    joblib.dump(tweet_prep, PREP_DUMP_FILEPATH)
    
    tweet_classifier = TweetClassifier(architecture=ARCHITECTURE,
                                       max_nr_words=MAX_WORDS,
                                       sequence_length=MAX_SEQ_LEN,
                                       embedding_dim=EMBEDDING_DIM,
                                       path_to_word_embeddings=EMBEDDING_MODEL_FILEPATH,
                                       word_index = tweet_prep.tokenizer.word_index,
                                       classes=2,
                                       model_save_filepath=CLASS_DUMP_FILEPATH)       
    
    tweet_classifier.train((X_train, y_train), (X_test, y_test), batch_size=128, epochs=5, verbose=2)