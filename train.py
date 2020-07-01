import argparse

import nrrd
import torch

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from models.Networks import ShapeEncoder

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def main(config):
    hyper_parameters = config['hyper_parameters']
    dirs = config['directories']

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(dirs['tensorboard'], config['name'], stats, hyper_parameters)
    
    # TODO:
    
    image_dir = 'data/nrrd_256_filter_div_32_solid/43321568c4bc0a7cbaf2e78ed413860a/43321568c4bc0a7cbaf2e78ed413860a.nrrd'
    data, _ = nrrd.read(image_dir, index_order='C')
    
    #[bs, in_c, depth, height, width]

    data = torch.randn(64, 4, 32, 32, 32) 
    model = ShapeEncoder()
    output = model(data)
    print(model)


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)