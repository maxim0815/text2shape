import argparse

import nrrd
import torch

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from dataloader.TripletLoader import TripletLoader

from dataloader.TextDataVectorization import TxtVectorization

from models.Networks import ShapeEncoder
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

    desc_batch = torch.zeros(2, 96).long()

    desc_batch[0] = torch.from_numpy(batch[0].pos_desc).long()
    desc_batch[1] = torch.from_numpy(batch[1].pos_desc).long()

    output = txt_encoder(desc_batch)

    print(txt_encoder)

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(
        dirs['tensorboard'], config['name'], stats, hyper_parameters)

    # TODO:
    
    image_dir = 'test/test_shapes/35bcb52fea44850bb97ad864945165a1/35bcb52fea44850bb97ad864945165a1.nrrd'
    data, _ = nrrd.read(image_dir, index_order='C')
    
    #[bs, in_c, depth, height, width]

    data = torch.randn(64, 4, 32, 32, 32) 
    shape_encoder = ShapeEncoder()
    output = shape_encoder(data)
    print(shape_encoder)


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
