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

    best_eval_loss = float('inf')
    best_ndcg = 0.0

    for ep in range(epochs):
        print("...starting with epoch {} of {}".format(ep, epochs))

        # TRAIN
        number_of_batches = int(
            dataloader.train_data.get_shape_length()/dataloader.bs)
        epoch_train_dict = eval_dict = {"loss": 0.0, "accuracy": 0.0}
        for i in range(number_of_batches):
            print('TRAIN: input {} of {} '.format(
                i, number_of_batches), end='\r')

            batch = dataloader.get_train_batch("s2t")
            train_dict = trip_enc.update(batch)
            epoch_train_dict["loss"] += train_dict["loss"]
            epoch_train_dict["accuracy"] += train_dict["accuracy"]

        train_dict = {"loss": epoch_train_dict["loss"]/number_of_batches,
                      "accuracy": epoch_train_dict["accuracy"]/number_of_batches}
        tensorboard.write_episode_data(ep, train_dict)

        # EVAL
        number_of_batches = int(
            dataloader.test_data.get_shape_length()/dataloader.bs)
        epoch_eval_dict = {"loss": 0.0, "accuracy": 0.0, "ndcg": 0.0}
        for i in range(number_of_batches):
            print('EVAL: input {} of {} '.format(
                i, number_of_batches), end='\r')

            batch = dataloader.get_test_batch("s2t")
            eval_dict = trip_enc.predict(batch)
            epoch_eval_dict["loss"] += eval_dict["loss"]
            epoch_eval_dict["accuracy"] += eval_dict["accuracy"]

        # run on metric
        if config['metric'] == "s2t":
            rand = np.random.randint(
                0, dataloader.test_data.get_shape_length())
            rand_shape = dataloader.test_data.get_shape(rand)
            closest_idx, _ = find_nn_shape_2_text(trip_enc.shape_encoder,
                                                  trip_enc.text_encoder,
                                                  rand_shape,
                                                  dataloader.test_data,
                                                  8)
            ndcg = calculate_ndcg(
                closest_idx, rand, dataloader.test_data, 8, "s2t")

        eval_dict = {"loss": epoch_eval_dict["loss"]/number_of_batches,
                     "accuracy": epoch_eval_dict["accuracy"]/number_of_batches,
                     "ndcg": ndcg}

        tensorboard_eval.write_episode_data(ep, eval_dict)

        if eval_dict["ndcg"] < best_ndcg:
            best_ndcg = eval_dict["ndcg"]
            print(
                "...new best eval ndcg {} --> saving models".format(best_ndcg))
            trip_enc.save_models()

    print("FINISHED")


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
