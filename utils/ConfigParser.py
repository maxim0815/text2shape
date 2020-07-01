import yaml

def config_parser(config_file):
	with open(config_file, "r") as ymlfile: 
		cfg = yaml.load(ymlfile)

	return cfg 