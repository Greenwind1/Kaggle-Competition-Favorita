# dipanjanS NLP blog


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
from keras.utils import np_utils
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib as plt

SEED = 2018
pd.options.display.max_colwidth = 200

np.random.seed(SEED)

import keras.backend as K
from keras.models import Sequential
from keras.layers import Embedding, Dense, Lambda

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
word2id['pad'] = 0
id2word = {v: k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in
        norm_bible]

vocab_size = len(word2id)
embed_size = 100
window_size = 2

# generate skip structure
i = 0
for x, y in generate_context(corpus=wids,
                             window_size=window_size,
                             vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Skip structure : ',
              [id2word[w] for w in x[0]],
              '-> Target : ',
              id2word[np.argwhere(y[0])[0][0]])
        if i == 10:
            break
        i += 1

cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size,
                   output_dim=embed_size,
                   input_length=window_size * 2))
# CBOW takes the average of all contexts
# ( In this case, average 4 words in contexts. )
cbow.add(Lambda(lambda encoded: K.mean(encoded, axis=1),
                output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(optimizer='rmsprop',
             loss='categorical_crossentropy')

for epoch in range(1, 6, 1):
    loss = 0.
    i = 0
    for x, y in generate_context(corpus=wids,
                                 window_size=window_size,
                                 vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100000 == 0:
            print('Processed {} context and word pairs'.format(i))

    print('Epoch : ', epoch, '\tLoss : ', loss, '\n')

cbow_config = cbow.get_config()

# with open('./cbow.pkl', 'wb') as f:
#     pickle.dump(cbow_config, f)
#     cbow_config = pickle.load(f)

weights = cbow.get_weights()[0]
weights = weights[1:]
cbow_embed = pd.DataFrame(weights, index=list(id2word.values())[:-1])
cbow_embed.head(2)

dist_matrix = euclidean_distances(weights)
sim_words = {
    i: [id2word[idx] for idx in dist_matrix[word2id[i] - 1].argsort()[1:6] + 1]
    for i in ['god', 'jesus']}
print(sim_words)
