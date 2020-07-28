[![Codacy Badge](https://api.codacy.com/project/badge/Grade/bf077e8ef1c64c1da2a5f4e804f86b62)](https://app.codacy.com/manual/maxim0815/text2shape?utm_source=github.com&utm_medium=referral&utm_content=maxim0815/text2shape&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/maxim0815/text2shape.svg?branch=master)](https://travis-ci.org/maxim0815/text2shape)

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


### Retrievals

* define which retrievals and further configs within config/cfg_retrieval.yaml
* possibile retrievals:
  * text 2 text   (t2t)
  * text 2 shape  (t2s)
  * shape 2 text  (s2t)
  * shape 2 shape (s2s)

```
python3 retrieval.py config/cfg_retrieval.yaml
```