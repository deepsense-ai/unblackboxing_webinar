import pandas as pd

from sklearn.manifold import TSNE

from unboxer.bokeh_tsne.utils import *
from unboxer.bokeh_tsne.hover_scatter import scatterplot_vis, scatterplot_text


class TsneBasic(object):
    def __init__(self, model, **tsne_kwargs):
        self.tsne_model_ = TSNE(**tsne_kwargs)
        self.model_ = model       
        self.tsne_features_ = None 

    def plot(self, **kwargs):
        pass
        
    def fit(self):
        pass
        

class TsneVis(TsneBasic):
    
    def __init__(self, model, feature_layer_name, **tsne_kwargs):
        super(TsneVis, self).__init__(model, **tsne_kwargs)
        self.feature_layer_name_ = feature_layer_name
        
    def plot(self, **kwargs):
        scatterplot_vis(self.tsne_features_, **kwargs)
    
    def fit(self, img_folder, label_df=pd.DataFrame(), batch_size=2):
        img_input_shape = self.model_.input_shape[1:-1]
        img_paths, img_tensor = folder2tensor(img_folder, paths=True, shape=None)
        img_features = self._extract_features(img_tensor, batch_size)
        
        tsne_features = self.tsne_model_.fit_transform(img_features)
        
        df = pd.DataFrame(tsne_features, columns=['x','y'])
        df['img_filepath'] = img_paths
        df.sort_values('img_filepath', inplace=True)

        if label_df.empty:
            df['label'] = 0
        else:
            label_df.sort_values('img_filepath', inplace=True)
            df.reset_index(inplace=True)
            label_df.reset_index(inplace=True)
            df['img_filepath_wtf'] = label_df['img_filepath']
            df['label'] = label_df['label']  
                      
        self.tsne_features_ = df
        
    def _extract_features(self,X, batch_size):
        layer_id = [i for i, l in enumerate(self.model_.layers) 
                    if l.name == self.feature_layer_name_][0]
        img_features = get_layer_output(self.model_, 
                                        layer=layer_id, X=X, batch_size=batch_size)

        new_shape = np.product(img_features.shape[1:])
        nr_imgs = img_features.shape[0]
        img_features = np.reshape(img_features,(nr_imgs, new_shape))
        return img_features
    
    
class TsneText(TsneBasic):
               
    def plot(self, **kwargs):
        scatterplot_text(self.tsne_features_, **kwargs)
    
    def fit(self, word_corpus, highlight_words=None):
        word_features = self._extract_features(word_corpus)
        
        tsne_features = self.tsne_model_.fit_transform(word_features)
        
        df = pd.DataFrame(tsne_features, columns=['x','y'])
        df['text'] = word_corpus
        df.sort_values('text', inplace=True)

        def highlight(x):
            if x in highlight_words:
                return 1
            else:
                return 0
        if highlight_words:
            df['label'] = df['text'].apply(highlight)
        else:
            df['label'] = 0
        self.tsne_features_ = df
        
    def _extract_features(self, X):
        vectors = []
        for w in X:
            w_id = self.model_.dictionary[w]
            vectors.append(self.model_.word_vectors[w_id])
        return np.stack(vectors, axis=0)