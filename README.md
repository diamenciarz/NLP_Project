# DACS NLP project - 2023 Group 9

In this project we are using arXiv dataset to determine paper category
and predict possible year of issue based on its text.

We obtain training and test data using publicly available 
[Kaggle dataset from Cornell University](https://www.kaggle.com/datasets/Cornell-University/arxiv).

# Filesystem structure

| Path  | Description                                         |
|:------|:----------------------------------------------------|
| /data | Location for downloaded and intermediate datasets   |
| /src  | Location for helper libraries and jupyter notebooks |
| /venv | Location for the python virtual environment         | 

# Initialization

To start working with the project, you have to follow several classic
steps.

1. Download and extract archive into `data` directory (so you would 
have `data/arxiv-metadata-oai-snapshot.json` file). 
2. Install virtualenv in `venv` directory.
3. Activate it.
4. Run `pip install -r requirements.txt` to install all dependencies..
5. Run `python src/setup.py` to set up missing pieces.
6. You're good to go!

# Development

`src/shared.py` contains the functions that can be reused across 
notebooks, including text normalization. Additional info is