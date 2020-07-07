import yaml


def pretty_config_print(config):
    print('_' * 37 + "CONFIG" + '_'*37)
    for key, val in config.items():
        if type(val) is dict:
            for k, v in val.items():
                print(k + '.' * (80 - len(k) - len(str(v))) + str(v))
        else:
            print(key + '.' * (80 - len(key) - len(str(val))) + str(val))
    print(80 * '_')


def config_parser(config_file, print_config=True):
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
	if 'ep' not in hp_:
		raise Exception("Check config file - ep not given")

	if 'data' not in dir_:
		raise Exception("Check config file - data dir not given")
	if 'output' not in dir_:
		raise Exception("Check config file - output dir not given")
	if 'model' not in dir_:
		raise Exception("Check config file - model dir not given")
	if 'tensorboard' not in dir_:
		raise Exception("Check config file - tensorboard dir not given")

	if print_config:
		pretty_config_print(cfg)

	return cfg 