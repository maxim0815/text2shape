import argparse
from utils.ConfigParser import config_parser

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="dir to config file")
    args = parser.parse_args()
    return args

def main(config):
	hyper_parameters = config['hyper_parameters']
	directories = config['directories']
	
	# TODO:


if __name__ == '__main__':
    args = parse_arguments()
    config = config_parser(args.config)
    main(config)