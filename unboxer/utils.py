import glob
from copy import deepcopy
import json

import numpy as np
from random import choice
from scipy.misc import imresize
from matplotlib import pyplot as plt
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from keras.preprocessing import image 
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras import backend as K
from keras import activations
from keras.utils.data_utils import get_file
from vis.utils.utils import find_layer_idx, apply_modifications

def load_query_image(filepath, tensor=False):
    img = image.load_img(filepath)
    img = image.img_to_array(img)
    if tensor:
        img = np.expand_dims(img, axis=0)
    return img

def prep_model_for_vis(model, out_layer_name='predictions'):
    layer_idx = find_layer_idx(model, out_layer_name)

    model.layers[layer_idx].activation = activations.linear
    model = apply_modifications(model)
    return model

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    
def get_images_from_directory(dir_path,extensions = ['jpg','jpeg','png']):
    '''
        Extracts filepaths of all the images with specified extensions from a folder
        
        Input:
            dir_path: string, full directory path
            extensions = list, list of valid extensions
        Output:
            list of valid filepaths
    '''
    
    types = ['{}/*.{}'.format(dir_path,e) for e in extensions]

    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    
    return files_grabbed

def get_pred_text_label(pred_id):
    CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    fpath = get_file('imagenet_class_index.json',
                 CLASS_INDEX_PATH,
                 cache_subdir='models')
    label_dict = json.load(open(fpath))
    return label_dict[str(pred_id)][1]

def img2tensor(img, img_shape):
    '''
        Transforms and preprocesses image for vgg model
        Inputs:
            img: numpy array, rgb image
        Outputs:
            tensor
    '''
    img = imresize(img, img_shape)
    img = image.img_to_array(img)    
    img = np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    
    return img

def deprocess_image(x, prep_mode = "imagenet"):
    _,img_height,img_width,img_channels = x.shape
    x = deepcopy(x)
    x = x.reshape((img_height, img_width, img_channels))
    
    if prep_mode =="imagenet":
        # Remove zero-center by mean pixel
        x[:, :, 0] += 104
        x[:, :, 1] += 117
        x[:, :, 2] += 124
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
    elif prep_mode == "simple":
        pass
    else:
        raise ValueError("wrong prep mode")
    return x

def folder2tensor(folder,
                  extensions = ['jpg','jpeg','png'],
                  paths = False,
                  img_shape=(224,224)
                 ):
    
    '''
        Reads and transforms a folder of images to tensor for keras models
        Inputs:
            folder: string, filepath to folder
            extensions:list, list of valid extensions
            mode: string, model mode leave as "imagenet" for now
            paths:boolean, whether paths should be outputed or only tensor
        Outputs:
            tensor build from the folder images or tuple list of (filepaths,tensor)
    '''
    
    img_paths = get_images_from_directory(folder,extensions)
    
    tensor_list = [img2tensor(plt.imread(im_pth)[:,:,:3], img_shape) 
                   for im_pth in img_paths]
    
    if paths:
        return img_paths,np.vstack(tensor_list)
    else:
        return np.vstack(tensor_list)

def load_model(img_size = (100,100,3),mode = "VGG"):
    '''softmax
        Helper for loading model. Only VGG available now
    '''
    
    if mode == "VGG":
        input_template = Input(batch_shape=(None,) + img_size)

        model = VGG16(input_tensor=input_template, 
                      weights='imagenet', 
                      include_top=False)
    else:
        raise NotImplemented
        
    return model

def get_activations(model, layer, X_batch):
    '''
        Gets activation outputs from a given model on a given layer for a chosen batch of images
        Inputs:
            model: keras model, image classification model
            layer: int or string, desired layer number or name 
            X_batch: tensor, batch of images on which outputs should be calculated
        Outputs:
            tensor of outputs from specified layer on a X_batch
    '''
    get_activations = K.function([model.layers[0].input, K.learning_phase()], 
                                 [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations[0]  
    
def normalize(x):
    '''
        Normalizes values from numpy array to [0,1]
        Input:
            x: numpy array, array to be normalized
    '''
    x += x.min()
    if x.max() == 0:
        return x
    else:
        x /= x.max()
        return x
    
def resize_folder(folder,
                  extensions = ['jpg','jpeg','png'],
                  size = (100,100)):
    
    '''
        Resizes all images in a specified folder to given size and overwrites all images in a folder
        Inputs:
            folder: string, full path to the folder with images
            extensions: list, list of valid extensions
            size: tuple, size of the output images
    '''
    
    img_paths = get_images_from_directory(folder,extensions)
    
    resized_imgs = [imresize(plt.imread(im_pth)[:,:,:3],size) for im_pth in img_paths]
    
    for p,r_im in zip(img_paths,resized_imgs):
        plt.imsave(p,r_im)
    
def random_image_crop(image,
                      size = (10,10),
                      nr = 20,
                      rotate = None
                     ):
    '''
        Extracts k random crops from an image of size specified
        
        Inputs:
            image: numpy array, image to be cropped
            size: tuple, shape of the crops
            nr: number of random crops
            rotate:boolean, if the image should be randomly rotated by i*90
        Ouputs:
            list of cropped images
    '''

    h,w = image.shape[:2]
    h_size,w_size = size

    crop_list = []
    for _ in range(nr):
    
        if rotate:
            image = np.rot90(image,k=choice(range(3)))
            h,w = image.shape[:2]
            h_size,w_size = size
        
        w_permitted = range(w-w_size) 
        h_permitted = range(h-h_size) 

        w_start = choice(w_permitted)
        w_end = w_start + w_size

        h_start = choice(h_permitted)
        h_end = h_start + h_size
        
        try:
            crop = image[h_start:h_end,w_start:w_end,:]
        except Exception:
            crop = image[h_start:h_end,w_start:w_end]
        
        crop_list.append(crop)
    return crop_list

def plot_folder(folder,extensions = ['jpg','jpeg','png','bmp'],**kwargs):
    '''
        Plots all the images from a specified folder with specified extensions
        
        Inputs:
            folder: string, path to folder with images
            extensions:list, list of valid extensions
    '''
    
    filepaths = get_images_from_directory(dir_path = folder,extensions = extensions)
    img_list = [plt.imread(f) for f in filepaths]
    plot_list(img_list,**kwargs)

def plot_list(img_list,labels = None, cols_nr = None):
    '''
        Plots a list of images in a grid with labels if specified
        Inputs:
            img_list: list, list of images to be plotted
            labels:list/array or None, labels to be added to images
            cols_nr: int or None, number of columns of the image grid
    '''
        
    n = len(img_list)
    
    if not cols_nr:
        cols_nr = int(np.sqrt(n))
    rows_nr = np.ceil(1.0*n/cols_nr)

    if labels is not None:       
        for i,(img,lab) in enumerate(zip(img_list,labels)):
            plt.subplot(rows_nr,cols_nr,i+1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(lab,fontsize=11)
            
        plt.tight_layout()
        plt.show()    
    else:
        for i,img in enumerate(img_list):
            plt.subplot(rows_nr,cols_nr,i+1)
            plt.imshow(img)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show() 