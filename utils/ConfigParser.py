import yaml

def print_symbol():
    print()
    print(19*' ' + '  / / / / / / / /   / / /         / / /  ' + 19*' ')
    print(19*' ' + '       / /       / /     / /   / /    / /' + 19*' ')
    print(19*' ' + '      / /             / /       / /      ' + 19*' ')
    print(19*' ' + '     / /           / /           / /     ' + 19*' ')
    print(19*' ' + '    / /         / /        / /    / /    ' + 19*' ')
    print(19*'_' + '___/_/_______/_/_/_/_/_/_____/_/_/_______' + 20*'_')
    print()

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
    if 'generate_batch' not in cfg:
        raise Exception("Check config file - No generate_batch within config file")
    if 'triplet' not in cfg:
        raise Exception("Check config file - No triplet within config file")
    if 'metric' not in cfg:
        raise Exception("Check config file - No metric within config file")
    if len(cfg['metric']) > 4:
        raise Exception("Check config file - More than four metrices within config file")
    if 'nns' not in cfg:
        raise Exception("Check config file - No nns within config file")
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
    if 'oversample' not in hp_:
        raise Exception("Check config file - oversample not given")


    if 'train_data' not in dir_:
        raise Exception("Check config file - train_data dir not given")
    if 'train_labels' not in dir_:
        raise Exception("Check config file - train_labels dir not given")
    if 'vocabulary' not in dir_:
        raise Exception("Check config file - vocabulary dir not given")
    if 'text_model_load' not in dir_:
        raise Exception("Check config file - text_model_load dir not given")
    if 'shape_model_load' not in dir_:
        raise Exception("Check config file - shape_model_load dir not given")
    if 'model_save' not in dir_:
        raise Exception("Check config file - model_save dir not given")
    if 'tensorboard' not in dir_:
        raise Exception("Check config file - tensorboard dir not given")

    if print_config:
        pretty_config_print(cfg)

    return cfg

def retrieval_config_parser(config_file, print_config=True):
    with open(config_file, "r") as ymlfile: 
        cfg = yaml.load(ymlfile)

    # check if all data are given in config file
    if 'name' not in cfg:
        raise Exception("Check config file - No name within config file")
    if 'hyper_parameters' not in cfg:
        raise Exception("Check config file - No hyper parameters within config file")
    if 'version' not in cfg:
        raise Exception("Check config file - No version within config file")	
    if 'directories' not in cfg:
        raise Exception("Check config file - No directories within config file")

    hp_ = cfg.get('hyper_parameters')
    dir_ = cfg.get('directories')

    if 'k' not in hp_:
        raise Exception("Check config file - k not given")

    if 'train_data' not in dir_:
        raise Exception("Check config file - train_data dir not given")
    if 'train_labels' not in dir_:
        raise Exception("Check config file - train_labels dir not given")
    if 'vocabulary' not in dir_:
        raise Exception("Check config file - vocabulary dir not given")
    if 'text_model_load' not in dir_:
        raise Exception("Check config file - text_model_load dir not given")
    if 'shape_model_load' not in dir_:
        raise Exception("Check config file - shape_model_load dir not given")
    if 'output' not in dir_:
        raise Exception("Check config file - output dir not given")
    
    print_symbol()
    if print_config:
        pretty_config_print(cfg)

    return cfg 