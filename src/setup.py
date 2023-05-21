import os
import nltk

seed = 42
SEED_ENVIRONMENT_VARIABLE = 'NLP_PROJECT_SEED'

if os.environ[SEED_ENVIRONMENT_VARIABLE] is not None:
    seed = int(os.environ[SEED_ENVIRONMENT_VARIABLE])

if __name__ == '__main__':
    nltk.download('stopwords')