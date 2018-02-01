import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from keras.utils.np_utils import to_categorical

from scipy.misc import imresize
from skimage.color import rgb2gray 


def lfw_train_test_split(lfw_people, **kwargs):
    images = lfw_people.images
    targets = lfw_people.target
    X_train, X_test, y_train, y_test = train_test_split(images,targets, **kwargs)
    return (X_train, y_train),(X_test, y_test)


class FacePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_classes_ = None
    
    def fit(self, X, y=None):
        if y is None:
            return self
        else:
            self.num_classes_ = len(set(y))
            return self

    def transform(self, X, y=None, copy=None):
        X /= 255.
        if y is None:
            return X
        else:
            y = to_categorical(y, self.num_classes_)
            return X, y  
    
    def fit_transform(self, X, y=None, copy=None):
        self.fit(X,y)        
        return self.transform(X, y)  
    
def tensor2img(tensor):
    return np.squeeze(tensor)*255.

def img2tensor(img, shape=(125,94)):
    img = imresize(img, shape)
    img = np.expand_dims(img,axis=0)
    return img
