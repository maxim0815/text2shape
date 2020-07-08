import argparse

import nrrd
import torch

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from dataloader.TripletLoader import TripletLoader

from dataloader.TextDataVectorization import TxtVectorization

from models.TripletEncoder import TripletEncoder

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def main(config):
    hyper_parameters = config['hyper_parameters']
    dirs = config['directories']

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(
        dirs['tensorboard'], config['name'], stats, hyper_parameters)

    dataloader = TripletLoader(config)

    trip_enc = TripletEncoder(config, dataloader.length_voc)

    epochs = config['hyper_parameters']['ep']

    print("...starting training")
    for ep in range(epochs):
        print("...starting with epoch {} of {}".format(ep, epochs))
        bs = config['hyper_parameters']['bs']
        length = dataloader.get_length("train")
        number_of_batches = int(dataloader.get_length("train")/config['hyper_parameters']['bs'])
        for i in range(number_of_batches):
            batch = dataloader.get_batch("train")
            eval_dict = trip_enc.update(batch)

            if i % 10 == 0:
                if ep == 0:
                    episode = i
                else:
                    episode = ep*i
                tensorboard.write_episode_data(episode, eval_dict)
            
            print('Input {} of {} '.format(i, number_of_batches), end='\r')



if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
