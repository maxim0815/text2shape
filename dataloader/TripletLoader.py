import torch.utils.data as data
import nrrd
import pandas as pd

import sys
import os


class TripletLoader(data.Dataset):
    def __init__(self, config):
        self.bs = config['hyper_parameters']['bs']

        try:
            self.descriptions = pd.read_csv(
                config['directories']['train_labels']).to_dict()
        except:
            sys.exit("ERROR! Triplet loader can't load given labels")

        try:
            self.shapes = parse_directory_for_nrrd(config['directories']['train_data'])
            #self.train_data, _ = nrrd.read(config['directories']['train_data'], index_order = 'C')
        except:
            sys.exit("ERROR! Triplet loader can't load given data")

        print("HUI")


def parse_directory_for_nrrd(path):
    shapes = dict()
    shapes['modelId'] = []
    shapes['data'] = []
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".nrrd"):
                train_data, _ = nrrd.read(
                    os.path.join(root, file), index_order='C')
                shapes['modelId'].append(file.replace('.nrrd', ''))
                shapes['data'].append(train_data)
    
    return shapes