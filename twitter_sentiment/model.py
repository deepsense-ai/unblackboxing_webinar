import numpy as np

from keras.layers import Conv1D, MaxPooling1D, Input, Dense, Flatten, Embedding, LSTM

from experiment.callbacks import BatchEndCallback, EpochEndCallback, ModelCheckpoint


class TweetClassifier(object):
    def __init__(self, architecture, max_nr_words, sequence_length, embedding_dim, path_to_word_embeddings, 
                 word_index, classes, model_save_filepath):
        
        self.model_save_filepath = model_save_filepath
        
        self.max_nr_words = max_nr_words
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.path_to_word_embeddings = path_to_word_embeddings       
    
        embedding_layer = self._make_embedding_layer(word_index)
    
        tweetnet = architecture(embedding_layer, sequence_length, classes)

        self.tweetnet = tweetnet
        self.tweetnet.summary()     

    def train(self, train, valid, batch_size, **kwargs):
        X_train, y_train = train
        X_valid, y_valid = valid

        checkpoint = ModelCheckpoint(filepath=self.model_save_filepath)
        batch_end_callback = BatchEndCallback()
        epoch_end_callback = EpochEndCallback()
        
        self.tweetnet.fit(X_train, y_train, 
                          validation_data=[X_valid, y_valid],
                          callbacks=[batch_end_callback, epoch_end_callback, checkpoint],
                          batch_size=batch_size, **kwargs)  

    def _make_embedding_layer(self, word_index):
        print('building embedding layer')
        embeddings = self._get_embeddings()
        nb_words = min(self.max_nr_words, len(word_index))
        embedding_matrix = np.zeros((nb_words, self.embedding_dim))
        
        for word, i in word_index.items():
            if i >= self.max_nr_words:
                continue
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(nb_words, self.embedding_dim, 
                                    weights=[embedding_matrix], 
                                    input_length=self.sequence_length, trainable=False)
        return embedding_layer

    def _get_embeddings(self):
        embeddings = {}
        with open(self.path_to_word_embeddings, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings[word] = coefs
        return embeddings