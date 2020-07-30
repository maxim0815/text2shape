import argparse
import random
import numpy as np

import torch
import torch.nn as nn

from models.Networks import T2SDiscriminator, T2SGenerator, TextEncoder

from dataloader.DataLoader import TextLoader

from utils.ConfigParser import gan_config_parser


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args


def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader = TextLoader(config)

    # Load text_encoder
    text_encoder = TextEncoder(loader.length_voc).to(device)
    temp_net = torch.load(
        config['directories']['text_model_load'], map_location=device)
    text_encoder.load_state_dict(temp_net)

    # GAN stuff
    generator = T2SGenerator()
    descriminator = T2SDiscriminator()

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(descriminator.parameters(), lr=0.001)

    loss = nn.BCELoss()
    epochs = config['hyper_parameters']['ep']
    for ep in range(epochs):
        generator_optimizer.zero_grad()

        # fake batch to satify batchnorm layer
        bs = 3
        desc_batch = torch.zeros((bs, 96)).long()
        description = random.choice(loader.descriptions['description'])
        desc_batch[0] = torch.from_numpy(description).long()
        description = random.choice(loader.descriptions['description'])
        desc_batch[1] = torch.from_numpy(description).long() 
        description = random.choice(loader.descriptions['description'])
        desc_batch[2] = torch.from_numpy(description).long()

        out = text_encoder(desc_batch)

        out_gen = generator(out)



if __name__ == '__main__':
    args = parse_arguments()
    config = gan_config_parser(args.config, print_config=True)
    main(config)
