# -*- coding:utf-8 -*-

from keras.preprocessing import sequence
from keras.utils import np_utils


def generate_context(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            # add surrounding words
            context_words.append([words[i]
                                  for i in range(start, end)
                                  if 0 <= i < sentence_length
                                  and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words,
                                       maxlen=context_length,
                                       value=0)
            y = np_utils.to_categorical(label_word,
                                        num_classes=vocab_size)
            yield (x, y)
