from nltk import *
from nltk.tokenize import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.preprocessing import sequence,text
from keras.models import *
from keras import *
from keras.layers import *
import keras
import os
import string
from nltk.corpus import stopwords
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = "ner"
Wk = word_tokenize
LEM = stem.WordNetLemmatizer()
EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 20
HIDDEN_LAYER_SIZE = 200
LAYERS = 1

def embedding(data):
    embeddings_index = {}
    f = open(os.path.join('glove.6B.50d.txt'))
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    for sent in data:
        for word in sent:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                embeddings_index[word] = np.random.rand(EMBEDDING_DIM)
                print("iiiiiiiiiiiiiiiiiiiiiiii  = ",word, " sss ")
    ind_to_word = {}
    word_to_ind = {}
    ind_to_vec = np.random.rand(len(embeddings_index)+1,EMBEDDING_DIM)
    ind_to_vec[0] = np.zeros(EMBEDDING_DIM)
    ind = 1
    for word , vec in embeddings_index.items():
        ind_to_word[ind] = word
        ind_to_vec[ind] = vec
        word_to_ind[word] = ind
        ind += 1
    return ind_to_vec , word_to_ind , ind_to_word

def seq_data(data,word_to_ind):
    temp = data.apply(lambda row: [word_to_ind[i] for i in row])
    return temp

def read_data():
    #message,food,recharge,support,reminders,travel,nearby,movies,casual,other
    train = pd.read_csv('ner_data')
    x = train['message']
    train_y = pd.read_csv('ner_label')
    y = train_y['message']
    x = preprocess(x)
    y = preprocess(y)
    dic = {"O":0,"date":1,"time":2,"location":3,"person":4,"ordinal":5,"gpe":6}
    new_y = []
    unique_label = ["O","date","time","location","person","ordinal","gpe"]
    for line in y:
        list_ = []
        for i in line:
            if i in unique_label:
                list_.append(dic[i])
            else:
                list_.append(dic["O"])
        new_y.append(list_)
    final_y = []
    for x_vector in new_y:
        x_rev_vector = []
        for index in x_vector:
            char_vector = np.zeros(len(dic))
            char_vector[index] = 1
            x_rev_vector.append(char_vector)
        final_y.append(np.asarray(x_rev_vector))
    return x ,final_y,unique_label

def preprocess(data,stem = False):
    #stop = stopwords.words('english')
    #print(stop)
    stop = list(string.punctuation)
    #print(string.punctuation)
    tokenizer = TreebankWordTokenizer()
    p_stemmer = PorterStemmer()
    list_of_X = data.apply(lambda row: row.lower())
    #list_of_X = list_of_X.apply(lambda row: [i for i in (row.split())])
    list_of_X = list_of_X.apply(lambda row: tokenizer.tokenize(row))
    #list_of_X = list_of_X.apply(lambda row: [LEM.lemmatize(i) for i in row])
    #list_of_X = list_of_X.apply(lambda row: [p_stemmer.stem(i) for i in row])
    list_of_X = list_of_X.apply(lambda row: [i for i in row if i not in stop])
    #list_of_X = list_of_X.apply(lambda row: str(row))
    return list_of_X

x , y ,unique_label = read_data()

ind_to_vec,word_to_ind,ind_to_word = embedding(x)

x_seq = seq_data(x , word_to_ind)
x_seq = keras.preprocessing.sequence.pad_sequences(x_seq,MAX_SEQUENCE_LENGTH,padding='pre',truncating='post',value=0)
y_seq = keras.preprocessing.sequence.pad_sequences(y,MAX_SEQUENCE_LENGTH,padding='pre',truncating='post',value=[1,0,0,0,0,0,0])

train_x, test_x, train_y, test_y = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

def KERAS():
    model = Sequential()
    model.add(Embedding(input_dim=len(ind_to_vec), output_dim=EMBEDDING_DIM,
                      weights=[ind_to_vec], input_length=MAX_SEQUENCE_LENGTH))

    model.add(Bidirectional(LSTM(64,unroll=True,return_sequences=True)))

    model.add(Dense(len(unique_label),activation = 'softmax'))

    model.compile(keras.optimizers.adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(np.array(train_x), np.array(train_y),epochs=20,batch_size=8,verbose=1,validation_data=(test_x,test_y))
   # pred = model.predict_classes(test_x)

    [print(n.name) for n in K.get_session().graph.as_graph_def().node]
KERAS()
