from utils.RenderShape import RenderImage
import torch
import argparse
import yaml
import os
import numpy as np

from utils.ConfigParser import retrieval_config_parser
from models.Networks import TextEncoder, ShapeEncoder
from dataloader.DataLoader import RetrievalLoader
from utils.NearestNeighbor import find_nn_text_2_text, find_nn_text_2_shape, \
    find_nn_shape_2_shape, find_nn_shape_2_text, \
    calculate_ndcg

#################################################################
# TODO:
#
#   Maybe later for using retrivals as metric for evaluation
#   during training TripletLoader and RetrievalLoder needs to be
#   fused
#
#   Parallelization for increasing loading time?
#################################################################


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def main(config):
    load_directory = []
    load_directory.append(config['directories']['shape_model_load'])
    load_directory.append(config['directories']['text_model_load'])

    # TODO: so far this is not necessary since loader base class
    #       functions are sufficent to load required data
    dataloader = RetrievalLoader(config)

    k = config["hyper_parameters"]["k"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    retrieval_versions = config["version"]

    ndcg_dict = {}

    for version in retrieval_versions:
        # text 2 text retrieval
        if version == "t2t":
            print(80 * '_')
            print("Running t2t retrieval ...")
            text_encoder = TextEncoder(dataloader.length_voc)
            temp_net = torch.load(load_directory[1], map_location=device)
            text_encoder = text_encoder.to(device)
            text_encoder.load_state_dict(temp_net)

            ndcg_list = []

            for n in range(config["hyper_parameters"]["n"]):
                # this is the description for which the nearest neighbors are searched
                rand = np.random.randint(
                    0, dataloader.get_description_length())
                rand_desc = dataloader.get_description(rand)

                closest_idx, closest_dist = find_nn_text_2_text(text_encoder,
                                                                rand_desc,
                                                                dataloader,
                                                                k)

                ndcg = calculate_ndcg(closest_idx, rand, dataloader, k, "t2t")
                ndcg_list.append(ndcg)
                print("...NDCG score : {:.2f}".format(ndcg))

                rand_desc = rand_desc.reshape(96)
                rand_desc = dataloader.txt_vectorization.vector2description(
                    rand_desc)

                nearest_descriptions = []

                for idx in closest_idx:
                    d = dataloader.get_description(idx)
                    d = d.reshape(96)
                    nearest_descriptions.append(
                        dataloader.txt_vectorization.vector2description(d))

                dict_ = {rand_desc: nearest_descriptions}

                save_directory = config["directories"]["output"]
                name = "t2t_nearest_neighbor_" + str(n) + ".yaml"
                file_name = os.path.join(save_directory, name)

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                with open(file_name, 'w+') as outfile:
                    yaml.dump(dict_, outfile, default_flow_style=False)

                print("...dumped file {} of {}".format(
                    n, config["hyper_parameters"]["n"]))
            
            ndcg_dict[version] = ndcg_list

        # text 2 shape retrieval
        if version == "t2s":
            print(80 * '_')
            print("Running t2s retrieval ...")
            shape_encoder = ShapeEncoder()
            temp_net = torch.load(load_directory[0], map_location=device)
            shape_encoder = shape_encoder.to(device)
            shape_encoder.load_state_dict(temp_net)

            text_encoder = TextEncoder(dataloader.length_voc)
            temp_net = torch.load(load_directory[1], map_location=device)
            text_encoder = text_encoder.to(device)
            text_encoder.load_state_dict(temp_net)

            ndcg_list = []

            for n in range(config["hyper_parameters"]["n"]):
                rand = np.random.randint(
                    0, dataloader.get_description_length())
                rand_desc = dataloader.get_description(rand)

                closest_idx, closest_dist = find_nn_text_2_shape(text_encoder,
                                                                 shape_encoder,
                                                                 rand_desc,
                                                                 dataloader,
                                                                 k)

                ndcg = calculate_ndcg(closest_idx, rand, dataloader, k, "t2s")
                ndcg_list.append(ndcg)
                print("...NDCG score : {:.2f}".format(ndcg))

                save_directory = config["directories"]["output"]
                folder = "text2shape" + str(n) + str("/")
                save_directory = os.path.join(save_directory, folder)
                file_name = os.path.join(save_directory, "descripton.yaml")

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # write description into yaml
                rand_desc = rand_desc.reshape(96)
                rand_desc = dataloader.txt_vectorization.vector2description(
                    rand_desc)

                dict_ = {"description": rand_desc}

                with open(file_name, 'w+') as outfile:
                    yaml.dump(dict_, outfile, default_flow_style=False)

                for idx in closest_idx:
                    n_shape = dataloader.get_shape(idx)
                    n_shape = n_shape.reshape(32, 32, 32, 4)
                    render = RenderImage()
                    render.set_shape(n_shape)
                    render.set_name(str(idx))
                    render.render_voxels(save_directory)

                print("...dumped pngs {} of {}".format(
                    n, config["hyper_parameters"]["n"]))
            
            ndcg_dict[version] = ndcg_list


        # shape 2 shape retrieval
        if version == "s2s":
            print(80 * '_')
            print("Running s2s retrieval ...")
            shape_encoder = ShapeEncoder()
            temp_net = torch.load(load_directory[0], map_location=device)
            shape_encoder = shape_encoder.to(device)
            shape_encoder.load_state_dict(temp_net)

            ndcg_list = []

            for n in range(config["hyper_parameters"]["n"]):
                # this is the shape for which the nearest neighbors are searched
                rand = np.random.randint(0, dataloader.get_shape_length())
                rand_shape = dataloader.get_shape(rand)

                closest_idx, closest_dist = find_nn_shape_2_shape(shape_encoder,
                                                                  rand_shape, 
                                                                  dataloader, 
                                                                  k)
                
                ndcg = calculate_ndcg(closest_idx, rand, dataloader, k, "s2s")
                ndcg_list.append(ndcg)
                print("...NDCG score : {:.2f}".format(ndcg))

                save_directory = config["directories"]["output"]
                name = "shape2shape" + str(n) + str("/")
                save_directory = os.path.join(save_directory, name)

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # save png of selected shape
                rand_shape = rand_shape.astype(int)
                rand_shape = rand_shape.reshape(32, 32, 32, 4)
                render = RenderImage()
                render.set_shape(rand_shape)
                render.set_name("selected")
                render.render_voxels(save_directory)

                for idx in closest_idx:
                    n_shape = dataloader.get_shape(idx)
                    n_shape = n_shape.reshape(32, 32, 32, 4)
                    render = RenderImage()
                    render.set_shape(n_shape)
                    render.set_name(str(idx))
                    render.render_voxels(save_directory)

                print("...dumped pngs {} of {}".format(
                    n, config["hyper_parameters"]["n"]))

            ndcg_dict[version] = ndcg_list

        # shape 2 text retrieval
        if version == "s2t":
            print(80 * '_')
            print("Running s2t retrieval ...")
            shape_encoder = ShapeEncoder()
            temp_net = torch.load(load_directory[0], map_location=device)
            shape_encoder = shape_encoder.to(device)
            shape_encoder.load_state_dict(temp_net)

            text_encoder = TextEncoder(dataloader.length_voc)
            temp_net = torch.load(load_directory[1], map_location=device)
            text_encoder = text_encoder.to(device)
            text_encoder.load_state_dict(temp_net)

            ndcg_list = []

            for n in range(config["hyper_parameters"]["n"]):
                rand = np.random.randint(0, dataloader.get_shape_length())
                rand_shape = dataloader.get_shape(rand)

                closest_idx, closest_dist = find_nn_shape_2_text(shape_encoder,
                                                                 text_encoder,
                                                                 rand_shape,
                                                                 dataloader,
                                                                 k)

                ndcg = calculate_ndcg(closest_idx, rand, dataloader, k, "s2t")
                ndcg_list.append(ndcg)
                print("...NDCG score : {:.2f}".format(ndcg))

                save_directory = config["directories"]["output"]
                folder = "shape2text" + str(n) + str("/")
                save_directory = os.path.join(save_directory, folder)
                file_name = os.path.join(save_directory, "descripton.yaml")

                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                # save png of selected shape
                rand_shape = rand_shape.reshape(32, 32, 32, 4)
                render = RenderImage()
                render.set_shape(rand_shape)
                render.set_name("selected")
                render.render_voxels(save_directory)

                nearest_descriptions = []

                for idx in closest_idx:
                    d = dataloader.get_description(idx)
                    d = d.reshape(96)
                    nearest_descriptions.append(
                        dataloader.txt_vectorization.vector2description(d))

                dict_ = {"descriptions": nearest_descriptions}

                with open(file_name, 'w+') as outfile:
                    yaml.dump(dict_, outfile, default_flow_style=False)

                print("...dumped file {} of {}".format(
                    n, config["hyper_parameters"]["n"]))
            
            ndcg_dict[version] = ndcg_list

    print("...dumping ndcg score")
    save_directory = config["directories"]["output"]
    file_name = os.path.join(save_directory, "ndcg_scores.yaml")
    with open(file_name, 'w+') as outfile:
        yaml.dump(ndcg_dict, outfile, default_flow_style=False)
    
    print("Retrieval task finished")

if __name__ == '__main__':
    args = parse_arguments()
    config = retrieval_config_parser(args.config)
    main(config)
