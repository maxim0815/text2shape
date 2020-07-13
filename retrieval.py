import torch
import argparse
import yaml
import os

from utils.ConfigParser import retrieval_config_parser
from models.Networks import TextEncoder, ShapeEncoder
from dataloader.TripletLoader import TripletLoader
from utils.NearestNeighbor import find_nn
from utils.RenderShape import RenderImage


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def main(config):
    load_directory = []
    load_directory.append(config['directories']['shape_model_load'])
    load_directory.append(config['directories']['text_model_load'])

    dataloader = TripletLoader(config)

    k = config["hyper_parameters"]["k"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    retrieval_versions = config["version"]

    for version in retrieval_versions:
        # text 2 text retrieval
        if version == "t2t":
            print("Running t2t retrieval ...")
            text_encoder = TextEncoder(dataloader.length_voc)
            temp_net = torch.load(load_directory[1], map_location=device)
            text_encoder = text_encoder.to(device)
            text_encoder.load_state_dict(temp_net)

            # generate tensor list of all descriptions within triplets
            data_list = []
            for triplet in dataloader.triplet_list:
                tensor = torch.from_numpy(triplet.pos_desc).long().to(device)
                tensor = tensor.unsqueeze(0)
                data_list.append(tensor)

            for n in range(config["hyper_parameters"]["n"]):
                # this is the description for which the nearest neighbors are searched
                rand_triplet = dataloader.get_batch("all")[0]
                desc = torch.from_numpy(rand_triplet.pos_desc).long()
                desc = desc.unsqueeze(0)        # fake bs of one for network
                desc = desc.to(device)

                closest_idx, closest_dist = find_nn(
                    text_encoder, desc, data_list, k)

                desc = desc.numpy()
                desc = desc.reshape(96)
                desc = dataloader.txt_vectorization.vector2description(desc)

                nearest_descriptions = []

                for idx in closest_idx:
                    d = data_list[idx]
                    d = d.numpy().reshape(96)
                    nearest_descriptions.append(
                        dataloader.txt_vectorization.vector2description(d))

                dict_ = {desc: nearest_descriptions}

                save_directory = config["directories"]["output"]
                name = "t2t_nearest_neighbor_" + str(n) + ".yaml"
                file_name = os.path.join(save_directory, name)

                with open(file_name, 'w') as outfile:
                    yaml.dump(dict_, outfile, default_flow_style=False)

                print("...dumped file {} of {}".format(
                    n, config["hyper_parameters"]["n"]))

        # text 2 shape retrieval
        if version == "t2s":
            print("Running t2s retrieval ...")

        # shape 2 shape retrieval
        if version == "s2s":
            print("Running s2s retrieval ...")
            shape_encoder = ShapeEncoder()
            temp_net = torch.load(load_directory[0], map_location=device)
            shape_encoder = shape_encoder.to(device)
            shape_encoder.load_state_dict(temp_net)

            # generate tensor list of all shapes
            # do not use triplet_list!
            # triplet list contrains some shapes multiple times with different desc.
            # TODO:
            # GENERATING NEW LIST OF TENSORS IS VERY COSTLY IN RAM
            data_list = []
            for shape in dataloader.shapes["data"]:
                tensor = torch.from_numpy(shape).float().to(device)
                tensor = tensor.unsqueeze(0)
                data_list.append(tensor)

            for n in range(config["hyper_parameters"]["n"]):
                # this is the shape for which the nearest neighbors are searched
                rand_triplet = dataloader.get_batch("all")[0]
                shape = torch.from_numpy(rand_triplet.shape).float()
                shape = shape.unsqueeze(0)        # fake bs of one for network
                shape = shape.to(device)

                closest_idx, closest_dist = find_nn(
                    shape_encoder, shape, data_list, k)

                save_directory = config["directories"]["output"]
                name = "shape2shape" + str(n) + str("/")
                file_dir = os.path.join(save_directory, name)

                # save png of selected shape
                shape = shape.int()
                shape = shape.numpy().reshape(32, 32, 32, 4)
                render = RenderImage()
                render.set_shape(shape)
                render.set_name("selected")
                render.render_voxels(file_dir)

                for idx in closest_idx:
                    n_shape = data_list[idx]
                    n_shape = n_shape.int()
                    n_shape = n_shape.numpy().reshape(32, 32, 32 ,4)
                    render = RenderImage()
                    render.set_shape(n_shape)
                    render.set_name(str(idx))
                    render.render_voxels(file_dir)

                print("...dumped pngs {} of {}".format(
                    n, config["hyper_parameters"]["n"]))


if __name__ == '__main__':
    args = parse_arguments()
    config = retrieval_config_parser(args.config)
    main(config)
