'''
Author: Ziyu Chen
'''
import pandas as pd
import numpy as np
import regex
import gensim
import nltk
import gensim
import os
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

    
def process_text(file_path = './data/embed_train.csv', maxlen = 100):
    '''
    Description: Process text data to be feed into neural networks.
    file_path: Path to the evidence sentences file.
    maxlen: The length of the sentences to be padded to.
    '''

    # Load and format files
    document = pd.read_csv(file_path, index_col=None)
    dat = document[['Sentence', 'label']]

    lines = []
    for x in dat['Sentence']:
        lines.append(len(list(filter(str.strip, x.split('\n')))))

    dat = dat.assign(length=lines)

    articles = []
    for x in dat['Sentence']:
        for i in range(len(list(filter(str.strip, x.split('\n'))))):
            articles.append(x.split('\n')[i])

    tags = []
    for index, row in dat.iterrows():
        for i in range(row['length']):
            tags.append(row['label'])

    data = pd.DataFrame({'lines': articles, 'label': tags})

    # Clean text
    pattern = "[^a-z|A-Z|\s]|\\b\w{1}\\b"
    X = data['lines']
    X = pd.Series([regex.sub(pattern, '', x) for x in X])
    y = data['label']
    labels = tags

    y = tags
    X = data['lines']
    X = X.values.astype(str)
    X = [word_tokenize(x) for x in X]
    strings = [' '.join(x) for x in X]
    string = ' '.join(strings)
    n_words = len(set(string.split(' ')))

    # Tokenization
    tokenizer = Tokenizer(nb_words=n_words)
    tokenizer.fit_on_texts(articles)
    sequences = tokenizer.texts_to_sequences(articles)
    word_index = tokenizer.word_index

    # Padding sentences to the same length
    X = pad_sequences(sequences, maxlen=maxlen)
    y = np_utils.to_categorical(np.asarray(labels))

    return X, y, word_index



def load_weights(pretrained_model = './model/wiki.en.bin', self_trained_model = './model/fil9.bin', word_index = None):
    '''
    Description: Load pretrained and self train word2vec model
    ''' 
    if word_index is None:
        X, y, word_index = process_text(file_path='./data/embed_train.csv', maxlen =100)
    if 'wiki.en.bin' not in os.listdir('./model'):
        pretrained_model = './model/wiki.en.vec'
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model, binary=False)
        wv_model.save_word2vec_format('./model/wiki.en.bin',binary=True)
        wv_model = gensim.models.KeyedVectors.load_word2vec_format('./model/wiki.en.bin', binary=True)
    else:
        wv_model = gensim.models.KeyedVectors.load_word2vec_format(pretrained_model, binary=True)
        
    my_model = gensim.models.KeyedVectors.load_word2vec_format(self_trained_model, binary = True)
    vector_sizes1 = my_model.vector_size
    vector_sizes2 = wv_model.vector_size
    vector_size = vector_sizes1 + vector_sizes2
    
    embeddings_index = {}
    for i in range(len(wv_model.index2word)):
        word = wv_model.index2word[i]
        coefs = np.asarray(wv_model.vectors[i], dtype='float32')
        embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, vector_size))
    for word, i in word_index.items():
        wv_vec = embeddings_index.get(word)

        if wv_vec is not None and word in my_model.wv.vocab:
            embedding_vector = np.append(wv_vec, my_model[word])
        elif wv_vec is not None and word not in my_model.wv.vocab:
            embedding_vector = np.append(wv_vec, np.zeros((1, 100)))
        elif wv_vec is not None and word in my_model.wv.vocab:
            embedding_vector = np.append(np.zeros((1, 300)), my_model[word])

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

    
def process_new_data(file_path = './data/embed_train.csv', maxlen = 100):
    new_data = pd.read_csv(file_path)
    new_dat = new_data['Sentence']

    new_dat = new_dat[new_dat.notnull()]

    new_articles = []
    for x in new_dat:
        for i in range(len(list(filter(str.strip ,x.split('\n'))))):
            new_articles.append(x.split('\n')[i])

    new_tokenizer = Tokenizer(nb_words=550)
    new_tokenizer.fit_on_texts(new_articles)
    new_sequences = tokenizer.texts_to_sequences(new_articles)

    new_word_index = new_tokenizer.word_index
    print('Found %s unique tokens.' % len(new_word_index))

    new_datas = pad_sequences(new_sequences, maxlen=100)
    
    return new_datas

    
