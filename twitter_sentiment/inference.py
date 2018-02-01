import pandas as pd


class TweetPredictor(object):
    def __init__(self, tweet_preprocessor, tweet_classifier):
        self.tweet_preprocessor = tweet_preprocessor
        self.tweet_classifier = tweet_classifier
    
    def predict_proba(self,tweet):
        X = self.tweet_preprocessor.transform(tweet)
        y_pred = self.tweet_classifier.predict(X)
        return y_pred
    
    def predict(self, tweet, threshold=0.5):
        y_pred = self.predict_proba(tweet)
        df = pd.DataFrame(y_pred[:,1], columns=['score'])
        def give_label(x):
            if x > threshold:
                return 'positive'
            else:
                return 'negative'
        df['predicted_label'] = df['score'].apply(give_label)
        return df