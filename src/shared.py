import os
import re
import string

import nltk
import pandas as pd

banned = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')

PROJECT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _tokenize(text):
    text = text.lower()
    tokens = (token for token in nltk.tokenize.word_tokenize(text) if token not in banned)
    return (stemmer.stem(token) for token in tokens)

def to_bag_of_words(text):
    result = {}
    for token in _tokenize(text.translate(string.punctuation)):
        result.setdefault(token, 0)
        result[token] += 1
    return result

def normalize(text):
    return  ' '.join(_tokenize(text))

def extract_category_names(encoded):
    """
    Extracts full category names from the encoded list -
    lower-order classification within a field of science (including
    field of science), such as computer science / machine learning
    or chemistry / explosives.

    :param encoded: String of categories as they come in the metadata
    :return: Set of category names encoded in the provided argument.
    """
    return [decoded.strip() for decoded in re.split('\s+|\s*,\s*', encoded)]

def extract_category_groups(encoded):
    """
    Extracts category groups from the encoded list - higher-order
    classification that refers to a field of science, such as
    engineering, agriculture,
    chemistry and so on.

    :param encoded: String of categories as they come in the metadata
    :return: Set of category groups encoded in the provided argument.
    """
    return [category.split('.')[0] for category in extract_category_names(encoded)]

def extract_year(versions):
    """
    Extracts year of paper from the version array

    :param versions: Array of versions in arxiv metadata.
    :return: Year of publication as an integer
    """

    timestamp = versions[-1]['created']
    match = re.search('\d{4}', timestamp)
    if match:
        return int(match.group(0))

def get_data_set_location(name):
    if not name.endswith('.json'):
        name += '.json'
    return os.path.join(PROJECT_DIRECTORY, 'data', name)

def process(chunk):
    chunk['category_names'] = chunk['categories'].apply(extract_category_names)
    chunk['category_groups'] = chunk['categories'].apply(extract_category_groups)
    chunk['words'] = chunk['abstract'].apply(to_bag_of_words)
    return chunk

def load_dataset(name):
    reader = pd.read_json(get_data_set_location(name), chunksize = 2 ** 12, lines = True)
    return (process(chunk) for chunk in reader)

# for chunk in load_dataset('arxiv-metadata-oai-snapshot.json'):
#     print(chunk)