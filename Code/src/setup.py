import os
import re
import multiprocessing
import sys

import nltk
import pandas as pd
import numpy as np

from shared import iterate_dataset, get_dataset_location, to_bag_of_words, DATA_DIRECTORY

SEED_ENVIRONMENT_VARIABLE = 'NLP_PROJECT_SEED'
DEFAULT_SEED = 42

def extract_category_names(encoded):
    """
    Extracts full category names from the encoded list -
    lower-order classification within a field of science (including
    field of science), such as computer science / machine learning
    or chemistry / explosives.

    :param encoded: String of categories as they come in the metadata
    :return: Set of category names encoded in the provided argument.
    """
    return {decoded.strip() for decoded in re.split('\s+|\s*,\s*', encoded)}

def extract_category_groups(encoded):
    """
    Extracts category groups from the encoded list - higher-order
    classification that refers to a field of science, such as
    engineering, agriculture,
    chemistry and so on.

    :param encoded: String of categories as they come in the metadata
    :return: Set of category groups encoded in the provided argument.
    """
    return {category.split('.')[0] for category in extract_category_names(encoded)}

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

def _process(chunk):
    chunk['category_names'] = chunk['categories'].apply(extract_category_names)
    chunk['category_groups'] = chunk['categories'].apply(extract_category_groups)
    chunk['words'] = chunk['abstract'].apply(to_bag_of_words)
    return chunk

def _load_dataset(name):
    return (_process(chunk) for chunk in iterate_dataset(name))


def _clean_dataset(name: str):
    import glob

    for file in glob.glob(os.path.join(DATA_DIRECTORY, name + '*')):
        print('Removing ' + file)
        os.remove(file)

def _append_to_dataset(name: str, frame: pd.DataFrame):
    path = get_dataset_location(name)
    if 'random_score' in frame:
        frame = frame.drop(['random_score'], axis = 1)
    frame.to_json(path, orient = 'records', lines = True, mode = 'a')

def _split_chunk(name: str, chunk: pd.DataFrame, test_size: float, validation_size: float):
    training_boundary = 1.0 - validation_size - test_size
    validation_boundary = 1.0 - validation_size

    split = {
        'training': chunk[(chunk['random_score'] < training_boundary)],
        'validation': chunk[(chunk['random_score'] >= training_boundary) & (chunk['random_score'] < validation_boundary)],
        'testing': chunk[(chunk['random_score'] >= validation_boundary)],
    }

    for key, frame in split.items():
        _append_to_dataset('%s.%s' % (name, key), frame)


def prepare_datasets(dataset: str, seed: int | None = None, chunk_size = 2 ** 14, development_set_size = 2 ** 15, test_size: float = 0.2, validation_size: float = 0.1):
    if seed is not None:
        np.random.seed(seed)

    def to_category_frame(frame):
        return pd.DataFrame({
            'abstract': frame['abstract'],
            'category_names': frame['categories'].apply(extract_category_names),
            'category_groups': frame['categories'].apply(extract_category_groups),
            'random_score': frame['random_score']
        })

    def to_year_frame(frame):
        return pd.DataFrame({
            'abstract': frame['abstract'],
            'year': frame['versions'].apply(extract_year),
            'random_score': frame['random_score']
        })

    processing = {'category': to_category_frame, 'year': to_year_frame}

    for key in processing.keys():
        _clean_dataset(key)
        _clean_dataset('%s.%d' % (key, chunk_size))

    processed = 0
    for chunk in iterate_dataset(dataset, chunk_size = chunk_size):
        chunk['abstract'] = chunk['abstract'].apply(to_bag_of_words)
        chunk['random_score'] = np.random.random_sample(chunk.shape[0])
        for name, transformer in processing.items():

            transformed = transformer(chunk)
            _split_chunk(name, transformed, test_size, validation_size)
            if processed < development_set_size:
                _split_chunk('%s.%d' % (name, development_set_size), transformed, test_size, validation_size)
        processed += chunk_size
        print('%i records processed' % processed)

def prepare_vocabulary(name: str, source: str, chunk_size: int = 2 ** 14):
    print('Processing vocabulary')
    vocabulary = {}
    processed = 0
    for chunk in iterate_dataset(source, chunk_size = chunk_size):
        for record in chunk['abstract']:
            for word, count in record.items():
                vocabulary.setdefault(word, {'documents': 0, 'count': 0})
                vocabulary[word]['count'] += count
                vocabulary[word]['documents'] += 1
        processed += chunk_size
        print('%i records processed' % processed)

    print('Word count: %d' % len(vocabulary))
    print('Writing vocabulary to file')
    _append_to_dataset(name, pd.DataFrame(vocabulary.items(), columns = ['word', 'count']))
    print('All done!')


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')

    seed = DEFAULT_SEED
    if os.environ.get(SEED_ENVIRONMENT_VARIABLE) is not None:
        seed = int(os.environ[SEED_ENVIRONMENT_VARIABLE])

    prepare_datasets('arxiv-metadata-oai-snapshot', seed = seed)
    # we need to build vocabulary for training set only, ignoring test and validation sets
    _clean_dataset('vocabulary')
    prepare_vocabulary('vocabulary', 'year.training')
