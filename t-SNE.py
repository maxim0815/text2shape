import numpy as np
import argparse
import yaml
import os
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE

from utils.ConfigParser import tsne_config_parser
from models.Networks import TextEncoder, ShapeEncoder
from dataloader.DataLoader import RetrievalLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def text_encoder(X, dataloader, load_directory):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder = TextEncoder(dataloader.length_voc)
    temp_net = torch.load(load_directory, map_location=device)
    text_encoder = text_encoder.to(device)
    text_encoder.load_state_dict(temp_net)

    X = torch.Tensor(X).to(device=device).long()
    X = text_encoder.forward(X)
    return X

def shape_encoder(X, dataloader, load_directory):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shape_encoder = ShapeEncoder()
    temp_net = torch.load(load_directory, map_location=device)
    shape_encoder = shape_encoder.to(device)
    shape_encoder.load_state_dict(temp_net)

    X = shape_encoder.forward(X)
    return X

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    plt.plot(X[:,0], X[:,1], "ro")
    # plt.show()

    plt.savefig('tsne.png')

def main(config):
    load_directory = []
    load_directory.append(config['directories']['shape_model_load'])
    load_directory.append(config['directories']['text_model_load'])

    dataloader = RetrievalLoader(config)

    n = config["hyper_parameters"]["n"]

    descriptions = []

    rand = np.random.randint(0, dataloader.get_description_length(), n)
    for i in range(n):
        descriptions.append(dataloader.get_description(rand[i]).tolist()[0])
    descriptions = text_encoder(descriptions, dataloader, load_directory[1])
    descriptions = descriptions.cpu().detach().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(descriptions)
    plot_embedding(X_embedded, "t-SNE embedding")


if __name__ == '__main__':
    args = parse_arguments()
    config = tsne_config_parser(args.config)
    main(config)