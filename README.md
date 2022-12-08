# Sentiment-Analysis-IMDB

<p align="center">
  <img src="/notebooks/images/results.png" width="400" height="150" />
</p>

#### RoBERTa Embeddings Evaluation via k-Means Clustering

<p align="center">
  <img src="/notebooks/images/clustering.png" width="400" height="400" />
</p>

## Problem Statement

This is a binary text classification problem where we predict the sentiment of movie reviews as either positive or negative. The classes are balanced.

## XGBoost

It is trained on features aggregated from character-TFIDF and word-TFIDF. Character-TFIDF has been used to account for misspellings.

The XGBoost model minimizes a custom binary logistic objective and uses accuracy score as the evaluation metric.

The training phase includes validating the model to find the optimal number of boosting rounds with early stopping and sets the classification threshold to maximize the accuracy score on a validation set.

## BERT, RoBERTa

These are pre-trained large language models that are fine-tuned by placing a classifier head on top.

## Ensemble

This is an ensemble of XGBoost, BERT and RoBERTa based on majority voting.

## Setup Instructions

### Move into top-level directory
```
cd Sentiment-Analysis-IMDB
```

### Install environment
```
conda env create -f environment.yml
```

### Activate environment
```
conda activate sentiment-analysis
```

### Install package
```
pip install -e src/sentiment-analysis
```
Including the optional -e flag will install sentiment-analysis in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

### Fetch data
```
python -m sentiment_analysis fetch
```

### Download NLTK data
```
python -m nltk.downloader all
```

### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter kernel to run notebooks.
