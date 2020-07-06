import argparse

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from dataloader.TripletLoader import TripletLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def main(config):
    hyper_parameters = config['hyper_parameters']
    dirs = config['directories']

    dataloader = TripletLoader(config)

    batch = dataloader.get_batch()
    

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(dirs['tensorboard'], config['name'], stats, hyper_parameters)
    
    # TODO:


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)