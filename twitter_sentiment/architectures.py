from keras.layers import Conv1D, MaxPooling1D, Input, Dense, Flatten, Embedding, LSTM
from keras.layers import TimeDistributed, Activation, RepeatVector, Permute, Lambda, merge
from keras.models import Model
import keras.backend as K


def arch_lstm(embedding_layer, sequence_length, classes):    
    tweet_input = Input(shape=(sequence_length,), dtype='int32')        
    embedded_tweet = embedding_layer(tweet_input)
    x = LSTM(64, activation='sigmoid', inner_activation='hard_sigmoid', 
             return_sequences=True)(embedded_tweet)
    x = LSTM(128, activation='sigmoid', inner_activation='hard_sigmoid', 
             return_sequences=False)(x)
    x = Dense(256, activation='relu')(x)
    tweet_output = Dense(classes, activation='softmax', name='predictions')(x)      

    tweetnet = Model(tweet_input, tweet_output)
    tweetnet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])    
    return tweetnet

def arch_conv1d(embedding_layer, sequence_length, classes):    
    tweet_input = Input(shape=(sequence_length,), dtype='int32')        
    embedded_tweet = embedding_layer(tweet_input)
    x = Conv1D(64, 5, activation='relu')(embedded_tweet)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    tweet_output = Dense(classes, activation='softmax',name='predictions')(x)      

    tweetnet = Model(tweet_input, tweet_output)
    tweetnet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])    
    return tweetnet

def arch_attention(embedding_layer, sequence_length, classes):    
    tweet_input = Input(shape=(sequence_length,), dtype='int32')        
    embedded_tweet = embedding_layer(tweet_input)
    
    activations = LSTM(128, return_sequences=True, name='recurrent_layer')(embedded_tweet)
    
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(128)(attention)
    attention = Permute([2, 1], name='attention_layer')(attention)
    
    sent_representation = merge([activations, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1), name='merged_layer')(sent_representation)

    tweet_output = Dense(classes, activation='softmax', name='predictions')(sent_representation)      

    tweetnet = Model(tweet_input, tweet_output)
    tweetnet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])    
    return tweetnet

def arch_attention36(embedding_layer, sequence_length, classes):    
    tweet_input = Input(shape=(sequence_length,), dtype='int32')        
    embedded_tweet = embedding_layer(tweet_input)
    
    activations = LSTM(36, return_sequences=True, name='recurrent_layer')(embedded_tweet)
    
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(36)(attention)
    attention = Permute([2, 1], name='attention_layer')(attention)
    
    sent_representation = merge([activations, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1), name='merged_layer')(sent_representation)

    tweet_output = Dense(classes, activation='softmax', name='output_layer')(sent_representation)      

    tweetnet = Model(tweet_input, tweet_output)
    tweetnet.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])    
    return tweetnet