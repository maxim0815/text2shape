import argparse

import nrrd
import torch
import random

import numpy as np

from utils.ConfigParser import config_parser
from utils.TensorboardEvaluation import Evaluation
from utils.NearestNeighbor import find_nn_shape_2_text, find_nn_shape_2_shape, \
    find_nn_text_2_shape, find_nn_text_2_text, calculate_ndcg

from dataloader.DataLoader import TripletLoader

from dataloader.TextDataVectorization import TxtVectorization

from models.TripletEncoder import TripletEncoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def run_metric(metric_list, n_neighbors, dataloader, encoder):
    ndcg_scores = dict()
    for metric in metric_list:
        if metric == "s2t":
            ndcg = 0
            for i in range(3):
                rand = np.random.randint(
                    0, dataloader.test_data.get_shape_length())
                rand_shape = dataloader.test_data.get_shape(rand)
                closest_idx, _ = find_nn_shape_2_text(encoder.shape_encoder,
                                                    encoder.text_encoder,
                                                    rand_shape,
                                                    dataloader.test_data,
                                                    n_neighbors)
                ndcg += calculate_ndcg(
                    closest_idx, rand, dataloader.test_data, n_neighbors, "s2t")
            ndcg_scores["s2t_ndcg"] = ndcg/(i+1)

        if metric == "s2s":
            ndcg = 0
            for i in range(3):
                rand = np.random.randint(
                    0, dataloader.test_data.get_shape_length())
                rand_shape = dataloader.test_data.get_shape(rand)
                closest_idx, _ = find_nn_shape_2_shape(encoder.shape_encoder,
                                                    rand_shape,
                                                    dataloader.test_data,
                                                    n_neighbors)
                ndcg += calculate_ndcg(
                    closest_idx, rand, dataloader.test_data, n_neighbors, "s2s")
            ndcg_scores["s2s_ndcg"] = ndcg/(i+1)

        if metric == "t2t":
            ndcg = 0
            for i in range(3):
                rand = np.random.randint(
                    0, dataloader.test_data.get_description_length())
                rand_desc = dataloader.test_data.get_description(rand)
                closest_idx, _ = find_nn_text_2_text(encoder.text_encoder,
                                                    rand_desc,
                                                    dataloader.test_data,
                                                    n_neighbors)
                ndcg += calculate_ndcg(
                    closest_idx, rand, dataloader.test_data, n_neighbors, "t2t")
            ndcg_scores["t2t_ndcg"] = ndcg/(i+1)

        if metric == "t2s":
            ndcg = 0
            for i in range(3):
                rand = np.random.randint(
                    0, dataloader.test_data.get_description_length())
                rand_desc = dataloader.test_data.get_description(rand)
                closest_idx, _ = find_nn_text_2_shape(encoder.text_encoder,
                                                    encoder.shape_encoder,
                                                    rand_desc,
                                                    dataloader.test_data,
                                                    n_neighbors)
                ndcg += calculate_ndcg(
                    closest_idx, rand, dataloader.test_data, n_neighbors, "t2s")
            ndcg_scores["t2s_ndcg"] = ndcg/(i+1)

    return ndcg_scores


def better_ndcg_scores(ndcg_scores, best_ndcg_scores):
    larger_ndcg_scores = list()
    for key, _ in ndcg_scores.items():
        if ndcg_scores[key] >= best_ndcg_scores[key]:
            larger_ndcg_scores.append(1)
    if len(ndcg_scores) == 1:
        if len(larger_ndcg_scores) > 0:
            return True
        else:
            False
    if len(ndcg_scores) > 1:
        if len(larger_ndcg_scores) >= len(ndcg_scores):
            return True
        else:
            False


def main(config):
    hyper_parameters = config['hyper_parameters']
    dirs = config['directories']
    metric = config["metric"]

    stats = ["loss", "accuracy"]
    tensorboard = Evaluation(
        dirs['tensorboard'], config['name'], stats, hyper_parameters)

    stats_eval = ["loss", "accuracy"]
    best_ndcg_scores = dict()
    best_eval_loss = np.inf
    for met in metric:
        stats_eval.append(met+"_ndcg")
        best_ndcg_scores[met+"_ndcg"] = 0.0

    tensorboard_eval = Evaluation(
        dirs['tensorboard'], config['name']+"_eval", stats_eval)

    dataloader = TripletLoader(config)

    trip_enc = TripletEncoder(config, dataloader.length_voc)

    epochs = config['hyper_parameters']['ep']
    triplet_versions = config['triplet']

    print("...starting training")

    for ep in range(epochs):
        print("...starting with epoch {} of {}".format(ep, epochs))

        # TRAIN
        number_of_batches = int(
            dataloader.train_data.get_shape_length()/dataloader.bs)
        epoch_train_dict = eval_dict = {"loss": 0.0, "accuracy": 0.0}
        for i in range(number_of_batches):
            print('TRAIN: input {} of {} '.format(
                i, number_of_batches), end='\r')

            generate_batch = config['generate_batch']
            if generate_batch == "mixed":
                generate_list = ["random", "smart"]
                generate_batch = random.choice(generate_list)

            if generate_batch == "random":
                if config['generate_condition'] == "uni_modal":
                    version = random.choice(triplet_versions)
                    batch = dataloader.get_train_batch(version)
                    batch_2 = 0
                if config['generate_condition'] == "cross_modal":
                    batch = dataloader.get_train_batch(triplet_versions[0])
                    batch_2 = dataloader.get_train_batch(triplet_versions[1])
            if generate_batch == "smart":
                if config['generate_condition'] == "uni_modal":
                    version = random.choice(triplet_versions)
                    batch = dataloader.get_train_smart_batch(version)
                    batch_2 = 0
                if config['generate_condition'] == "cross_modal":
                    batch = dataloader.get_train_smart_batch(triplet_versions[0])
                    batch_2 = dataloader.get_train_smart_batch(triplet_versions[1])
    
            train_dict = trip_enc.update(batch, batch_2)
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
            
            generate_batch = config['generate_batch']
            if generate_batch == "mixed":
                generate_list = ["random", "smart"]
                generate_batch = random.choice(generate_list)

            if generate_batch == "random":
                if config['generate_condition'] == "uni_modal":
                    version = random.choice(triplet_versions)
                    batch = dataloader.get_test_batch(version)
                    batch_2 = 0
                if config['generate_condition'] == "cross_modal":
                    batch = dataloader.get_test_batch(triplet_versions[0])
                    batch_2 = dataloader.get_test_batch(triplet_versions[1])
            if generate_batch == "smart":
                if config['generate_condition'] == "uni_modal":
                    version = random.choice(triplet_versions)
                    batch = dataloader.get_test_smart_batch(version)
                    batch_2 = 0
                if config['generate_condition'] == "cross_modal":
                    batch = dataloader.get_test_smart_batch(triplet_versions[0])
                    batch_2 = dataloader.get_test_smart_batch(triplet_versions[1])

            eval_dict = trip_enc.predict(batch, batch_2)
            epoch_eval_dict["loss"] += eval_dict["loss"]
            epoch_eval_dict["accuracy"] += eval_dict["accuracy"]

        # run on metric
        ndcg_scores = run_metric(
            config["metric"], config['nns'], dataloader, trip_enc)

        eval_dict["loss"] = epoch_eval_dict["loss"]/number_of_batches
        eval_dict["accuracy"] = epoch_eval_dict["accuracy"]/number_of_batches

        for key, value in ndcg_scores.items():
            eval_dict[key] = value

        tensorboard_eval.write_episode_data(ep, eval_dict)

        # check if ndcg scores are better than before
        # all metrices musst be better than best one before

        if best_eval_loss > eval_dict['loss']:
            best_eval_loss = eval_dict['loss']
            print("...new best eval loss --> saving models")
            trip_enc.save_models()

        #if better_ndcg_scores(ndcg_scores, best_ndcg_scores):
        #    for key, _ in ndcg_scores.items():
        #        if ndcg_scores[key] >= best_ndcg_scores[key]:
        #            best_ndcg_scores[key] = ndcg_scores[key]
        #    print("...new best eval ndcg score(s) --> saving models")
        #    trip_enc.save_models()

    print("FINISHED")


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config, print_config=True)
    main(config)
