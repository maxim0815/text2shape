# text2shape

Project is based on this [paper](https://arxiv.org/abs/1803.08495).
___
## Setup

### Dataset

The dataset can be find [here](http://text2shape.stanford.edu/).
Download and add to data folder:
* Text Descriptions
* Solid Voxelizations: 32 Resolution

### Dependencies
* pynrrd 0.4.2 :
  ```
  pip install pynrrd
  ```
* spacy-2.3.0 :
  ```
  pip install spacy
  python3 -m spacy download en_core_web_sm
  ```

* language-tool-python-2.2.3 :
  ```
  pip install language-tool-python
  ```
___
## Getting started

### Preprocessing for descriptions

* remove descriptions with more than *max_length* words (default 96 words)
* preprocessing description (each word/symbol is seperated by space)
* vocabulary gets filled with words that appear more than twice

```
python3 preprocessing/run_preprocessing.py data/captions.tablechair.csv data/full_preprocessed.captions.csv data/full_voc.csv
```

### Learning embeddings

* set configuration in config/cfg.yaml

```
python3 train.py config/cfg.yaml
```