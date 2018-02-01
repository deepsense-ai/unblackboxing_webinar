import sys, os

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from IPython.html import widgets
from IPython.display import display
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from vis.visualization import visualize_activation
from vis.utils import utils  

from unboxer.utils import plot_list, prep_model_for_vis


class DeepVis():
    def __init__(self, model, save_dir):
        self.model_ = model
        self.layer_filter_ids_ = self._build_layer_filter_dict(self.model_)           
        self.save_dir_ = save_dir
    
    def browse(self, figsize=(16,10), labels=None):        
        def plot(layer_id, filter_id):
            filepath = '{}/{}/{}/img.jpg'.format(self.save_dir_, 
                                                 layer_id, filter_id)            
            img = plt.imread(filepath)
            plt.figure(figsize=figsize)
            if labels:
                plt.title('Label: {}'.format(labels[int(filter_id)]))
            plt.imshow(img)
            plt.show()
        return interact(plot, layer_id='1',filter_id='0')

    def browse_layer(self, batch_size=25, cols=5):
        def plot(layer_id, batch_id):
            plt.figure(figsize=(14,20))
            all_files = sorted(os.listdir('{}/{}'.format(self.save_dir_, layer_id)))
            
            batch_id = int(batch_id)
            img_list, label_list = [],[]
            for f in all_files[batch_id*batch_size:(batch_id+1)*batch_size]:
                img_path = os.path.join(self.save_dir_, layer_id, f, 'img.jpg')
                img = plt.imread(img_path)
                img_list.append(img)
                label_list.append(f)
            plot_list(img_list, label_list, cols_nr=cols)
        return interact(plot, layer_id='17',batch_id='6')

    def generate_max_activation_images(self, layer_ids):
    
        for layer_id in layer_ids:
            for filter_id in range(self.layer_filter_ids_[layer_id]):
                print('layer:{} filter:{}'.format(layer_id,filter_id))
                maximal_activation_image = self.find_mai(layer_id,filter_id)
                self.save(layer_id, filter_id, maximal_activation_image)
                
    def find_mai(self, layer_id, filter_id):        
        img = visualize_activation(model=self.model_, 
                                   layer_idx=layer_id, 
                                   filter_indices=[filter_id], 
                                   max_iter=500, verbose=False)
        return img
    
    def save(self, layer_id, filter_id, img):
        directory = '{}/{}/{}'.format(self.save_dir_, layer_id, filter_id)
        
        if not os.path.exists(directory): os.makedirs(directory)
        filepath = os.path.join(directory,'img.jpg')
        plt.imsave(filepath, img)
         
    def _build_layer_filter_dict(self, model):
        layer_filter_dict = {}
        for i,l in enumerate(model.layers):
            try:
                filter_shape = l.get_weights()[0].shape
            except Exception:
                continue
            layer_filter_dict[i] = filter_shape[-1]
        return layer_filter_dict
    
if __name__ == '__main__':
    
    deep_vis = DeepVis(model_architecture='vgg16', save_dir='/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/data/filter_images')
    conv_layer_ids = [11, 12, 13, 15, 16, 17]
    deep_vis.generate_max_activation_images(conv_layer_ids)