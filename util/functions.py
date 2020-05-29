import pandas as pd
import pickle


# pickle writer
def pickle_write(data, filename, byte=True):
    print('-' * 60)
    print('Writing...')
    if byte:
        with open(filename, 'wb') as fw:
            pickle.dump(data, fw)
        print('Writing has finised.')
    else:
        with open(filename, 'w') as fw:
            pickle.dump(data, fw)
        print('Writing has finised.')


# pickle reader
def pickle_read(filename, byte=True, verbose=True):
    if verbose: print('-' * 50 + '\nLoading...')
    if byte:
        with open(filename, 'rb') as fw:
            data = pickle.load(fw)
        if verbose: print('Loading has finised.')
        return data
    else:
        with open(filename, 'r') as fw:
            data = pickle.load(fw)
        if verbose: print('Loading has finised.')
        return data


# python 2
def to_unicode(unicode_or_str):
    if isinstance(unicode_or_str, str):
        value = unicode_or_str.decode('utf-8')
    else:
        value = unicode_or_str
    return value


def to_str(unicode_or_str):
    if isinstance(unicode_or_str, unicode):
        value = unicode_or_str.encode('utf-8')
    else:
        value = unicode_or_str
