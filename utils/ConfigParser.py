import yaml

def config_parser(config_file):
	with open(config_file, "r") as ymlfile: 
		cfg = yaml.load(ymlfile)

	# check if all data are given in config file
	if 'name' not in cfg:
		raise Exception("Check config file - No name within config file")
	if 'hyper_parameters' not in cfg:
		raise Exception("Check config file - No hyper parameters within config file")
	if 'directories' not in cfg:
		raise Exception("Check config file - No directories within config file")

	hp_ = cfg.get('hyper_parameters')
	dir_ = cfg.get('directories')

	if 'lr' not in hp_:
		raise Exception("Check config file - lr not given")
	if 'bs' not in hp_:
		raise Exception("Check config file - bs not given")
	if 'mom' not in hp_:
		raise Exception("Check config file - mom not given")
	if 'wd' not in hp_:
		raise Exception("Check config file - wd not given")

	if 'data' not in dir_:
		raise Exception("Check config file - data dir not given")
	if 'output' not in dir_:
		raise Exception("Check config file - output dir not given")
	if 'model' not in dir_:
		raise Exception("Check config file - model dir not given")
	if 'tensorboard' not in dir_:
		raise Exception("Check config file - tensorboard dir not given")

	return cfg 