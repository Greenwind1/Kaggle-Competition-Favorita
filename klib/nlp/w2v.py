# dipanjanS NLP blog
# Word2vec model ( Skip gram model )


import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk.corpus import gutenberg
from string import punctuation
from nlp_lib.normalizer import normalize_doc
from nlp_lib.skip_words import generate_context
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import skipgrams
from keras.utils import np_utils
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt

SEED = 2018
pd.options.display.max_colwidth = 200

np.random.seed(SEED)

import keras.backend as K
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Reshape
from keras.layers import Merge

bible = gutenberg.sents('bible-kjv.txt')
rm_terms = punctuation + '0123456789'

norm_bible = [[word.lower() for word in sent if word not in rm_terms] for sent
              in bible]
norm_bible = [' '.join(tok_sent) for tok_sent in norm_bible]
norm_bible = [normalize_doc(i) for i in norm_bible if not None]
norm_bible = [i for i in norm_bible if len(i.split()) > 2]

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(norm_bible)
word2id = tokenizer.word_index
id2word = {v: k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in
        norm_bible]

vocab_size = len(word2id) + 1
embed_size = 100

# generate skip-gram
skip_grams = [skipgrams(wid, vocab_size, window_size=10) for wid in wids]

# View sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
for i in range(len(pairs)):
    print('{}({}), {}({}) => {}'.format(id2word[pairs[i][0]],
                                        pairs[i][0],
                                        id2word[pairs[i][1]],
                                        pairs[i][1],
                                        labels[i]
                                        ))

# Build Skip-gram DNN
# -----------------------------------------------
w2v_word = Sequential()
w2v_word.add(Embedding(input_dim=vocab_size,
                       output_dim=embed_size,
                       embeddings_initializer='glorot_uniform',
                       input_length=1))
w2v_word.add(Reshape((embed_size,)))
# -----------------------------------------------
w2v_context = Sequential()
w2v_context.add(Embedding(input_dim=vocab_size,
                          output_dim=embed_size,
                          embeddings_initializer='glorot_uniform',
                          input_length=1))
w2v_context.add(Reshape((embed_size,)))
# -----------------------------------------------
w2v = Sequential()
w2v.add(Merge([w2v_word, w2v_context], mode='dot'))
w2v.add(Dense(units=1, activation='sigmoid'))
w2v.compile(optimizer='rmsprop',
            loss='mean_squared_error')
print(w2v.summary())

for epoch in range(1, 2, 1):
    loss = 0
    for i, elem in enumerate(skip_grams):
        pair_first_elem = np.array(list(zip(*elem[0]))[0],
                                   dtype='int32')
        pair_second_elem = np.array(list(zip(*elem[0]))[1],
                                    dtype='int32')
        labels = np.array(elem[1], dtype='int32')
        X = [pair_first_elem, pair_second_elem]
        y = labels
        if i % 1000 == 0:
            print('Processed {} skip gram pairs'.format(i))
        loss += w2v.train_on_batch(X, y)

    print('Epoch : ', epoch, '\tLoss : ', loss, '\n')

w2v_config = [w2v_word.get_config(),
              w2v_context.get_config(),
              w2v.get_config()]

# with open('./w2v.pkl', 'wb') as f:
#     pickle.dump(w2v_config, f)
#     w2v_config = pickle.load(f)

merge_layer = w2v.layers[0]
word_model = merge_layer.layers[0]
word_embed_layer = word_model.layers[0]
weights = word_embed_layer.get_weights()[0][1:]

w2v_embed = pd.DataFrame(weights, index=list(id2word.values()))
w2v_embed.head(2)

dist_matrix = euclidean_distances(weights)
sim_words = {
    i: [id2word[idx] for idx in dist_matrix[word2id[i] - 1].argsort()[1:6] + 1]
    for i in ['god', 'jesus']}
print(sim_words)
