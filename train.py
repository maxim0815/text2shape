import argparse

import nrrd
import torch

import numpy as np

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation
from utils.NearestNeighbor import find_nn_shape_2_text, calculate_ndcg

from dataloader.DataLoader import TripletLoader

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
    
    stats_eval = ["loss", "accuracy", "ndcg"]
    tensorboard_eval = Evaluation(
        dirs['tensorboard'], config['name']+"_eval", stats_eval)

    dataloader = TripletLoader(config)

    trip_enc = TripletEncoder(config, dataloader.length_voc)

    epochs = config['hyper_parameters']['ep']

    print("...starting training")

    episode = 0
    best_eval_loss = float('inf')

    for ep in range(epochs):
        print("...starting with epoch {} of {}".format(ep, epochs))

# TODO: this is bullshit --> look for smart way
        number_of_batches = int(
            dataloader.train_data.get_description_length()/dataloader.bs)
        for i in range(number_of_batches):
            print('Input {} of {} '.format(i, number_of_batches), end='\r')

            batch = dataloader.get_train_batch("s2t")
            train_dict = trip_enc.update(batch)

            # if episode % 10 == 0:
            tensorboard.write_episode_data(episode, train_dict)

        if episode % 10 == 0:
            # run evaluation on one batch
            batch_eval = dataloader.get_test_batch("s2t")

            # run on metric
            if config['metric'] == "s2t":
                rand = np.random.randint(0, dataloader.test_data.get_shape_length())
                rand_shape = dataloader.test_data.get_shape(rand)
                closest_idx, _ = find_nn_shape_2_text(trip_enc.shape_encoder,
                                                                    trip_enc.text_encoder,
                                                                    rand_shape,
                                                                    dataloader.test_data,
                                                                    8)
                ndcg = calculate_ndcg(closest_idx, rand, dataloader.test_data, 8, "s2t")
            
            eval_dict = trip_enc.predict(batch_eval)
            eval_dict['ndcg'] = ndcg

            tensorboard_eval.write_episode_data(episode, eval_dict)

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
