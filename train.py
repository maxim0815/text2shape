import argparse

import numpy as np

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from loss.MultimodalLoss import *

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

    #T = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    #S = torch.tensor([[6,1,0],[2,9,7],[1,4,2]])
    #T = torch.tensor([[1,2,3],[4,5,6],[7,8,9], [1,1,1]])
    #S = torch.tensor([[6,1,0],[2,9,7],[1,4,2], [1,1,1]])
    T = torch.tensor([[1,2,3,4],[4,5,6,6],[7,8,9,9]])
    S = torch.tensor([[6,1,0,6],[2,9,7,4],[1,4,2,4]])

    L_TST = cross_modal_association_loss(T, S)



if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)