import numpy as np
import argparse
import yaml
import os
import torch
import matplotlib.pyplot as plt

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE

from utils.RenderShape import RenderImage
from utils.ConfigParser import tsne_config_parser
from models.Networks import TextEncoder, ShapeEncoder
from dataloader.DataLoader import RetrievalLoader
from utils.NearestNeighbor import find_nn_text_2_text, find_nn_text_2_shape, \
    find_nn_shape_2_shape, find_nn_shape_2_text, \
    calculate_ndcg

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def find_positive_shape_id(desc_id, dataloader):
    matching_idx = [i for i, x in enumerate(
        dataloader.shapes['modelId']) if x == desc_id]
    if len(matching_idx) == 0:
        return None
    rand = np.random.randint(0, len(matching_idx))
    return matching_idx[rand]

def main(config):
    load_directory = config['directories']['text_model_load']

    dataloader = RetrievalLoader(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get pretrained network
    text_encoder = TextEncoder(dataloader.length_voc)
    temp_net = torch.load(load_directory, map_location=device)
    text_encoder = text_encoder.to(device)
    text_encoder.load_state_dict(temp_net)

    n = config["hyper_parameters"]["n"]
    descriptions = []
    rand = np.random.randint(0, dataloader.get_description_length(), n)
    for i in range(n):
        descriptions.append(dataloader.get_description(rand[i]).tolist()[0])

    X = torch.Tensor(descriptions).to(device=device).long()
    X = text_encoder.forward(X)

    X_encoded = X.cpu().detach().numpy()
    X_embedded = TSNE(n_components=2).fit_transform(X_encoded)

    x_min, x_max = np.min(X_embedded, 0), np.max(X_embedded, 0)
    X = (X_embedded - x_min) / (x_max - x_min)

    # plt.figure()
    # plt.rcParams["figure.figsize"] = (30,24)

    fig, ax = plt.subplots()

    save_directory = config["directories"]["output"]
    folder = "tsne_img" + str("/")
    save_directory = os.path.join(save_directory, folder)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for i in range(n):
        desc_id = dataloader.descriptions['modelId'][rand[i]]
        idx = find_positive_shape_id(desc_id, dataloader)
        if idx != None:
            print('plot {} of {} '.format(
                i, n), end='\r')
            # ax.plot(X[i,0], X[i,1], "ro")
            n_shape = dataloader.get_shape(idx)
            n_shape = n_shape.reshape(32, 32, 32, 4)
            render = RenderImage()
            render.set_shape(n_shape)
            render.set_name(str(idx))
            render.render_voxels(save_directory)

            img = plt.imread(save_directory+str(idx)+".png", format='png')
            img_cropped = img[40:400, 50:540, :]
            imagebox = OffsetImage(img_cropped, zoom=0.1)
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, (X[i,0], X[i,1]),
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.0)
            ax.add_artist(ab)

    # Fix the display limits to see everything
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # plt.show()

    plt.savefig('results/tsne.png')


if __name__ == '__main__':
    args = parse_arguments()
    config = tsne_config_parser(args.config)
    main(config)