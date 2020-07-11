import torch
import argparse
import yaml
import os

from utils.ConfigParser import retrieval_config_parser
from models.Networks import TextEncoder
from dataloader.TripletLoader import TripletLoader
from utils.NearestNeighbor import find_nn

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

    # text 2 text retrieval
    if config["version"] == "t2t":
        text_encoder = TextEncoder(dataloader.length_voc)
        temp_net = torch.load(load_directory[1], map_location=device)
        text_encoder = text_encoder.to(device)
        text_encoder.load_state_dict(temp_net)

        data_list = []
        for triplet in dataloader.triplet_list:
            tensor = torch.from_numpy(triplet.pos_desc).long().to(device)
            tensor = tensor.unsqueeze(0)
            data_list.append(tensor)

        for n in range(config["hyper_parameters"]["n"]):
            rand_triplet = dataloader.get_batch("all")[0]

            desc = torch.from_numpy(rand_triplet.pos_desc).long()
            desc = desc.unsqueeze(0)
            desc = desc.to(device)
            
            closest_idx, closest_dist = find_nn(text_encoder, desc, data_list, k)

            desc = desc.numpy()
            desc = desc.reshape(96)
            desc = dataloader.txt_vectorization.vector2description(desc)

            nearest_descriptions = []

            for idx in closest_idx:
                d = data_list[idx]
                d = d.numpy().reshape(96)
                nearest_descriptions.append(dataloader.txt_vectorization.vector2description(d))

            dict_ = {desc: nearest_descriptions}

            save_directory = config["directories"]["output"]
            name = "t2t_nearest_neighbor_" + str(n) + ".yaml"
            file_name = os.path.join(save_directory, name)

            with open(file_name, 'w') as outfile:
                yaml.dump(dict_, outfile, default_flow_style=False)
            
            print("HUI")


if __name__ == '__main__':
    args = parse_arguments()
    config = retrieval_config_parser(args.config)
    main(config)
