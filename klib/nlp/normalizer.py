# -*- coding:utf-8 -*-

import re
import nltk


def normalize_doc(doc):
    """
    sub = substitute
        pattern : special sequences(¥), special characters
                ( e.g. ¥s : unicode whitespace )
        repl : replacement ( string or function )
        string : words to be changed
    ^ = caret, matches the start of the string.
    re.I = ignore case
    re.A = make ¥w, ¥W, ..., ¥s and ¥S perform ASCII-only matching
    """

    wpt = nltk.WordPunctTokenizer()
    stop_words = nltk.corpus.stopwords.words('english')

    doc = re.sub(r'[^a-zA-Z¥s]', ' ', doc, re.I | re.A)
    # strip remove whitespace and return in default
    doc = doc.lower().strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if
                       token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc
