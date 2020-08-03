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

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    # plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)

    plt.savefig('tsne.png')

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
    plot_embedding(X_embedded, "t-SNE embedding")


if __name__ == '__main__':
    args = parse_arguments()
    config = retrieval_config_parser(args.config)
    main(config)