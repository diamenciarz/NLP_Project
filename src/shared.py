import os
import re
import string
from typing import Iterable

import nltk
import pandas as pd

PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'data')
STEMMER = nltk.stem.SnowballStemmer('english')
DELIMITERS = {'+', '-', '/', '='}

def _tokenize(text: str, stop_words = None) -> Iterable[str]:
    stop_words = set() if stop_words is None else stop_words
    text = text.lower()
    # for some reason abstracts are full of pluses instead of spaces
    for delimiter in DELIMITERS:
        text = text.replace(delimiter, ' ')
    tokens = (token for token in nltk.tokenize.word_tokenize(text) if token not in stop_words)
    return (STEMMER.stem(token) for token in tokens)

def to_bag_of_words(text):
    result = {}
    for token in _tokenize(text.translate(string.punctuation)):
        result.setdefault(token, 0)
        result[token] += 1
    return result

def normalize(text, stop_words = None):
    return  ' '.join(_tokenize(text, stop_words))

def get_dataset_location(name):
    if not name.endswith('.json'):
        name += '.json'
    return os.path.join(PROJECT_DIRECTORY, 'data', name)

def iterate_dataset(name, chunk_size = 2 ** 12):
    return pd.read_json(get_dataset_location(name), chunksize = chunk_size, lines = True)
