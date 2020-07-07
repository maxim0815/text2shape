import argparse

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from dataloader.TripletLoader import TripletLoader

from dataloader.TextDataVectorization import TxtVectorization

from models.Networks import TextEncoder

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def main(config):
    hyper_parameters = config['hyper_parameters']
    dirs = config['directories']

    vectorization = TxtVectorization(config['directories']['vocabulary'])

    s = "the table is round and has 3 legs . the table is rotating ."
    vec = vectorization.description2vector(s)
    des = vectorization.vector2description(vec)

    dataloader = TripletLoader(config)

    batch = dataloader.get_batch()

    pos_desc = dataloader.txt_vectorization.vector2description(
        batch[0].pos_desc)
    neg_desc = dataloader.txt_vectorization.vector2description(
        batch[0].neg_desc)

    txt_encoder = TextEncoder(dataloader.length_voc)

    desc_tensor = torch.from_numpy(batch[0].pos_desc).long()
    desc_tensor = desc_tensor.unsqueeze(0)              # simulates bs
    #desc_tensor = torch.LongTensor(desc_tensor)
    test = torch.randn(1, 4380) 
    # swap axis into [bs, n_class, len]
    #desc_tensor = desc_tensor.view(1, desc_tensor.shape[0], 100)

    output = txt_encoder(desc_tensor)

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(
        dirs['tensorboard'], config['name'], stats, hyper_parameters)

    # TODO:


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
