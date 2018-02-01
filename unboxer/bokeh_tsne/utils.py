import glob

import numpy as np
from random import choice

from matplotlib import pyplot as plt
import seaborn as sns

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras import backend as K

def get_layer_output(model, layer, X, batch_size=4):
    '''
    '''
    get_batch_output = K.function([model.layers[0].input, K.learning_phase()], 
                                 [model.layers[layer].output,])
    layer_output = []
    for i, X_batch in enumerate(batch(X, batch_size)):
        print('%s of %s'%((i+1)*batch_size,X.shape[0]))
        layer_output.append(get_batch_output([X_batch,0])[0])
    layer_output = np.vstack(layer_output) 
    return layer_output

def batch(X,batch_size):
    n = len(X)
    k = n%batch_size
    for start, stop in zip(range(0, n - batch_size + 1, batch_size), 
                           (range(batch_size, n + 1, batch_size))):
        yield X[start:stop]
    if k != 0:
        yield X[-k:]

def get_images_from_directory(dir_path,extensions = ['jpg','jpeg','png','bmp','BMP']):
    '''
    '''
    
    types = ['{}/*.{}'.format(dir_path,e) for e in extensions]

    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    
    return files_grabbed

def img2tensor(img):
    '''
    '''
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    
    return img

def folder2tensor(folder,
                  extensions=['jpg','jpeg','png','BMP'],
                  paths=False,
                  shape=None
                 ):
    
    '''
    '''
    
    img_paths = get_images_from_directory(folder,extensions)
    
    if shape:
        #resize here
        tensor_list = [img2tensor(plt.imread(im_pth)[:,:,:3]) 
                       for im_pth in img_paths]
    tensor_list = [img2tensor(plt.imread(im_pth)[:,:,:3]) for im_pth in img_paths]
    
    if paths:
        return img_paths,np.vstack(tensor_list)
    else:
        return np.vstack(tensor_list)