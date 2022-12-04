# Sentiment-Analysis-IMDB


## Setup Instructions

### Move into top-level directory
```
cd IMDB-Sentiment-Analysis
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

### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter kernel to run notebooks.
