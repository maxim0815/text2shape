import argparse

import nrrd
import torch

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation

from dataloader.TripletLoader import TripletLoader

from dataloader.TextDataVectorization import TxtVectorization

from models.TripletEncoder import TripletEncoder


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
    tensorboard_eval = Evaluation(
        dirs['tensorboard'], config['name']+"_eval", stats)

    dataloader = TripletLoader(config)

    trip_enc = TripletEncoder(config, dataloader.length_voc)

    epochs = config['hyper_parameters']['ep']

    print("...starting training")

    episode = 0
    best_eval_loss = float('inf') 
    best_eval_acc = 0.0

    for ep in range(epochs):
        print("...starting with epoch {} of {}".format(ep, epochs))
        bs = config['hyper_parameters']['bs']
        length = dataloader.get_length("train")
        number_of_batches = int(dataloader.get_length(
            "train")/config['hyper_parameters']['bs'])
        for i in range(number_of_batches):
            print('Input {} of {} '.format(i, number_of_batches), end='\r')

            batch = dataloader.get_batch("train")
            eval_dict = trip_enc.update(batch)

            #if episode % 10 == 0:
            tensorboard.write_episode_data(episode, eval_dict)

            if episode % 10 == 0:
                # run evaluation on one batch
                batch_eval = dataloader.get_batch("test")
                eval_dict = trip_enc.predict(batch_eval)

                tensorboard_eval.write_episode_data(episode, eval_dict)

                # run for loss
                # TODO: update maybe to accuracy?

                if eval_dict["loss"] < best_eval_loss:
                    best_eval_loss = eval_dict["loss"]
                    print(
                        "...new best eval Loss {} --> saving models".format(best_eval_loss))
                    trip_enc.save_models()

            episode += 1
    print("FINISHED")

if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
