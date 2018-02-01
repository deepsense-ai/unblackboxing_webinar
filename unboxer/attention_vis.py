import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact

import keras.backend as K

from twitter_sentiment.inference import TweetPredictor


class AttentionVisualizer(TweetPredictor):
    def __init__(self, tweet_processor, tweet_clasifier, max_layer_len):
        super(AttentionVisualizer, self).__init__(tweet_processor, tweet_clasifier)
        self.max_layer_len = max_layer_len
    
    def type_and_vis(self):
        def input_box(tweet, grads, activations, over_words, over_units):
            self.vis_activation([tweet], grads, activations, over_words, over_units)
        return interact(input_box, tweet='This is a great tool', grads=False, activations=True, 
                        over_words=True, over_units=False)
    
    def vis_activation(self, tweet, grads=False, activations=True,  over_words=True, over_units=False):  
        pred = self.predict(tweet)
        act_grad_matrix, layer_labels, text_labels = self._get_activations_gradients(tweet, grads, 
                                                                                     activations, over_words, over_units)

        plt.figure(figsize=(14,4))
        cmap = sns.diverging_palette(220, 20, n=7)
        ax = sns.heatmap(act_grad_matrix, xticklabels=text_labels, yticklabels=layer_labels, cmap=cmap)
        ax.xaxis.tick_top()
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        plt.title('Score:%s'%pred['score'].values[0])
        plt.show()
    
    def _get_activations_gradients(self, tweet, grads, activations, over_words, over_units):
        X = self.tweet_preprocessor.transform(tweet)
        functions = self._get_output_functions()
        outputs = functions([X,0])
        
        names = ['recurrent_layer','attention_layer','merged_layer', 'predictions']
        names = ['%s_%s'%(ni, nj) for ni in names for nj in ['activation','gradient']] 
        activations_gradients = {n:o for n,o in zip(names,outputs)}  
        
        name_labels = []
        act_grad_matrix = []
        for k,v in activations_gradients.items():
            if not grads:
                if 'gradient' in k:
                    continue
            if not activations:
                if 'activation' in k:
                    continue            
            if 'predictions' in k:
                continue
                
            v = np.squeeze(v)
            if 'recurrent' in k in k:
                v0 = v.mean(axis=0)
                v0 = self._assert_pad(v0, self.max_layer_len)
                v1 = v.mean(axis=1)
                v1 = self._assert_pad(v1, self.max_layer_len)

                if over_words:
                    name_labels.append(k+'_over_words')
                    act_grad_matrix.append(v0)
                if over_units:
                    name_labels.append(k+'_over_units')
                    act_grad_matrix.append(v1)
            else:
                name_labels.append(k)            
                v= self._assert_pad(v, self.max_layer_len)
                act_grad_matrix.append(v)
        act_grad_matrix = np.stack(act_grad_matrix, axis=0)
        
        text = tweet[0]
        text = text.split(' ')
        text_labels = [''] * self.max_layer_len
        for i, w in enumerate(text):
            text_labels[i] = w
            
        return act_grad_matrix, name_labels, text_labels
    
    def _get_output_functions(self):
        # if you name your layers you can use model.get_layer('recurrent_layer')
        model = self.tweet_classifier
        recurrent_layer = model.layers[2]
        attention_layer = model.layers[5]
        merged_layer = model.layers[9]
        output_layer = model.layers[10]
        layers = [recurrent_layer, attention_layer, merged_layer, output_layer]   
        
        outputs = []        
        for l in layers:
            outputs.append(l.output)

            loss = K.mean(model.output)
            grads = K.gradients(loss, l.output)
            grads_norm = grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5)
            outputs.append(grads_norm)

        all_function = K.function([model.layers[0].input, K.learning_phase()],
                                  outputs)
        return all_function
    
    def _assert_pad(self, v, max_layer_size):
        vs = v.shape[0]
        if vs < max_layer_size:
            v = np.pad(v, (0, max_layer_size - vs), mode='constant')
        return v