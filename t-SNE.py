import numpy as np
import argparse
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE

from utils.ConfigParser import retrieval_config_parser
from dataloader.DataLoader import RetrievalLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def main(config):
    load_directory = []
    load_directory.append(config['directories']['shape_model_load'])
    load_directory.append(config['directories']['text_model_load'])

    dataloader = RetrievalLoader(config)

    descriptions = []
    # for i in range(dataloader.get_description_length()):
    for i in range(5):
        descriptions.append(dataloader.get_description(i).tolist()[0])
    X_embedded = TSNE(n_components=2).fit_transform(descriptions)
    sns.scatterplot(x="X", y="Y", data=X_embedded)

if __name__ == '__main__':
    args = parse_arguments()
    config = retrieval_config_parser(args.config)
    main(config)