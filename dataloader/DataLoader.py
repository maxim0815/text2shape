import torch
import nrrd
import pandas as pd
import numpy as np
import random

import sys
import os

from dataloader.TextDataVectorization import TxtVectorization


class TripletShape2Text(object):
    def __init__(self, shape, pos_desc, neg_desc):
        self.shape = shape
        self.pos_desc = pos_desc
        self.neg_desc = neg_desc


class TripletText2Shape(object):
    def __init__(self, desc, pos_shape, neg_shape):
        self.desc = desc
        self.pos_shape = pos_shape
        self.neg_shape = neg_shape


class Loader(object):
    '''
    Loader class
        tries to load given files
        handles exceptions
        adds category to shape data
        converts dict{dict{}} to dict{list[]}
        codes all descriptions to vector
    '''

    def __init__(self, config):
        try:
            self.descriptions = pd.read_csv(
                config['directories']['train_labels']).to_dict()
        except:
            sys.exit("ERROR! Loader can't load given labels")

        try:
            self.shapes = parse_directory_for_nrrd(
                config['directories']['train_data'])
            #self.train_data, _ = nrrd.read(config['directories']['train_data'], index_order = 'C')
        except:
            sys.exit("ERROR! Loader can't load given data")

        try:
            self.txt_vectorization = TxtVectorization(
                config['directories']['vocabulary'])
        except:
            sys.exit("ERROR! Loader can't load given vocabulary")

        self.length_voc = len(self.txt_vectorization.voc_list)

        self.__add_category_to_shape()
        self.__description_to_lists()
        self.__description_to_vector()

    def __add_category_to_shape(self):
        """
        shape is needed for
            calculating the ndcg
            smart batches
        """
        category = list()
        for _, shape_id in enumerate(self.shapes['modelId']):
            cat = "none"
            found_category = False
            for key, value in self.descriptions["modelId"].items():
                if value == shape_id:
                    found_category = True
                    break
            if found_category == True:
                cat = self.descriptions['category'][key]
                category.append(cat)
        self.shapes['category'] = category

    def __description_to_lists(self):
        """
        pandas stores dict within dict which contains idx as key
        this functions converts these dicts into one list
        """
        for key, _ in self.descriptions.items():
            data_list = []
            for _, v in self.descriptions[key].items():
                data_list.append(v)
            self.descriptions[key] = data_list

    def __description_to_vector(self):
        desc_vector_list = list()
        for desc in self.descriptions["description"]:
            desc_vector_list.append(self.txt_vectorization.description2vector(
                desc))
        self.descriptions["description"] = desc_vector_list


class DataLoader(object):
    '''
    holds either:
        all data    -->     just retrieval
        train data or
        test data
    '''

    def __init__(self, descriptions, shapes):
        self.descriptions = descriptions
        self.shapes = shapes

    def get_shape_length(self):
        return len(self.shapes["modelId"])

    def get_description_length(self):
        return len(self.descriptions["modelId"])

    def get_shape(self, id):
        shape = self.shapes["data"][id]
        shape = np.expand_dims(shape, axis=0)           # bs = 1
        return shape

    def get_description(self, id):
        desc = self.descriptions["description"][id]
        desc = np.expand_dims(desc, axis=0)             # bs = 1
        return desc


class TripletLoader(object):
    """
    uses loader to get all data
    splits data into train and test DataLoader
    generates different triplet batches
    """

    def __init__(self, config):
        loader = Loader(config)

        # TODO: seed to config?
        np.random.seed(1200)

        self.bs = config['hyper_parameters']['bs']
        self.txt_vectorization = loader.txt_vectorization
        self.length_voc = len(self.txt_vectorization.voc_list)

        self.__split_train_test(loader)

    def __split_train_test(self, loader):
        '''
        split 80/20 pareto ratio
        first split shapes
        then look for matching descriptions for shapes and add to corresponding data container
        '''
        train_descriptions = {'id': list(), 'modelId': list(), 'description': list(), 'category': list(),
                              'topLevelSynsetId': list(), 'subSynsetId': list()}
        test_descriptions = {'id': list(), 'modelId': list(), 'description': list(), 'category': list(),
                             'topLevelSynsetId': list(), 'subSynsetId': list()}
        train_shapes = dict()
        test_shapes = dict()

        end_train = int(len(loader.shapes['modelId'])*0.8)
        for key, _ in loader.shapes.items():
            d1 = list(loader.shapes[key][:end_train])
            d2 = list(loader.shapes[key][end_train:])
            train_shapes[key] = d1
            test_shapes[key] = d2

        for i, shape_id in enumerate(train_shapes['modelId']):
            idx = [i for i, x in enumerate(
                loader.descriptions['modelId']) if x == shape_id]
            for key, val_list in loader.descriptions.items():
                for id in idx:
                    train_descriptions[key].append(val_list[id])

            print("Generate train split {} of {}".format(
                i, len(train_shapes['modelId'])), end='\r')
        print()
        for i, shape_id in enumerate(test_shapes['modelId']):
            idx = [i for i, x in enumerate(
                loader.descriptions['modelId']) if x == shape_id]
            for key, val_list in loader.descriptions.items():
                for id in idx:
                    test_descriptions[key].append(val_list[id])

            print("Generate test split {} of {}".format(
                i, len(test_shapes['modelId'])), end='\r')
        print()

        self.train_data = DataLoader(train_descriptions, train_shapes)
        self.test_data = DataLoader(test_descriptions, test_shapes)

    def get_train_batch(self, version):
        batch = []
        if version == "s2t":
            for _ in range(self.bs):
                rand = np.random.randint(0, self.train_data.get_shape_length())
                shape_id = self.train_data.shapes["modelId"][rand]
                shape = self.train_data.shapes['data'][rand]

                pos_id = self.__find_positive_description_id(shape_id)
                # in case of no matching positive description is found
                while pos_id == None:
                    rand = np.random.randint(
                        0, self.train_data.get_shape_length())
                    shape_id = self.train_data.shapes["modelId"][rand]
                    shape = self.train_data.shapes['data'][rand]

                    pos_id = self.__find_positive_description_id(shape_id)

                pos_desc = self.train_data.descriptions["description"][pos_id]

                neg_id = self.__find_negative_desciption_id(shape_id)
                neg_desc = self.train_data.descriptions["description"][neg_id]

                triplet = TripletShape2Text(shape, pos_desc, neg_desc)
                batch.append(triplet)
        if version == "t2s":
            for _ in range(self.bs):
                rand = np.random.randint(
                    0, self.train_data.get_description_length())
                desc_id = self.train_data.descriptions['modelId'][rand]
                desc = self.train_data.descriptions["description"][rand]

                pos_id = self.__find_positive_shape_id(desc_id, "train")

                # in case of no matching positive shape is found
                while pos_id == None:
                    rand = np.random.randint(
                        0, self.train_data.get_description_length())
                    desc_id = self.train_data.descriptions['modelId'][rand]
                    desc = self.train_data.descriptions["description"][rand]

                    pos_id = self.__find_positive_shape_id(desc_id)
                
                pos_shape = self.train_data.shapes['data'][pos_id]

                neg_id = self.__find_negative_shape_id(desc_id)
                neg_shape = self.train_data.shapes['data'][neg_id]

                triplet = TripletText2Shape(desc, pos_shape, neg_shape)
                batch.append(triplet)

        return batch

    def get_test_batch(self, version):
        batch = []
        if version == "s2t":
            for _ in range(self.bs):
                rand = np.random.randint(0, self.test_data.get_shape_length())
                shape_id = self.test_data.shapes["modelId"][rand]
                shape = self.test_data.shapes['data'][rand]

                pos_id = self.__find_positive_description_id(
                    shape_id, data="test")
                # in case of no matching positive description is found
                while pos_id == None:
                    rand = np.random.randint(
                        0, self.test_data.get_shape_length())
                    shape_id = self.test_data.shapes["modelId"][rand]
                    shape = self.test_data.shapes['data'][rand]

                    pos_id = self.__find_positive_description_id(
                        shape_id, data="test")

                pos_desc = self.test_data.descriptions["description"][pos_id]

                neg_id = self.__find_negative_desciption_id(
                    shape_id, data="test")
                neg_desc = self.test_data.descriptions["description"][neg_id]

                triplet = TripletShape2Text(shape, pos_desc, neg_desc)

                batch.append(triplet)
        return batch

    def __find_positive_description_id(self, shape_id, data="train"):
        """
        return random matching idx of all desciptions
        """
        if data == "train":
            matching_idx = [i for i, x in enumerate(
                self.train_data.descriptions['modelId']) if x == shape_id]
            if len(matching_idx) == 0:
                print("MISSING DESCIPTION FOR ID : {}".format(shape_id))
                return None
            rand = np.random.randint(0, len(matching_idx))
            return matching_idx[rand]
        if data == "test":
            matching_idx = [i for i, x in enumerate(
                self.test_data.descriptions['modelId']) if x == shape_id]
            if len(matching_idx) == 0:
                print("MISSING DESCIPTION FOR ID : {}".format(shape_id))
                return None
            rand = np.random.randint(0, len(matching_idx))
            return matching_idx[rand]

    def __find_negative_desciption_id(self, shape_id, data="train"):
        if data == "train":
            max_val = len(self.train_data.descriptions["modelId"])
            rand = np.random.randint(0, max_val)
            while self.train_data.descriptions["modelId"][rand] == shape_id:
                rand = np.random.randint(0, max_val)
            return rand
        if data == "test":
            max_val = len(self.test_data.descriptions["modelId"])
            rand = np.random.randint(0, max_val)
            while self.test_data.descriptions["modelId"][rand] == shape_id:
                rand = np.random.randint(0, max_val)
            return rand

    def __find_positive_shape_id(self, desc_id, data="train"):
        if data == "train":
            matching_idx = [i for i, x in enumerate(
                self.train_data.shapes['modelId']) if x == desc_id]
            if len(matching_idx) == 0:
                print("MISSING SHAPE FOR ID : {}".format(desc_id))
                return None
            rand = np.random.randint(0, len(matching_idx))
            return matching_idx[rand]
        if data == "test":
            matching_idx = [i for i, x in enumerate(
                self.test_data.shapes['modelId']) if x == desc_id]
            if len(matching_idx) == 0:
                print("MISSING SHAPE FOR ID : {}".format(desc_id))
                return None
            rand = np.random.randint(0, len(matching_idx))
            return matching_idx[rand]

    def __find_negative_shape_id(self, desc_id, data="train"):
        if data == "train":
            max_val = len(self.train_data.shapes["modelId"])
            rand = np.random.randint(0, max_val)
            while self.train_data.shapes["modelId"][rand] == desc_id:
                rand = np.random.randint(0, max_val)
            return rand
        if data == "test":
            max_val = len(self.test_data.shapes["modelId"])
            rand = np.random.randint(0, max_val)
            while self.test_data.shapes["modelId"][rand] == desc_id:
                rand = np.random.randint(0, max_val)
            return rand


class RetrievalLoader(DataLoader):
    """
    holds all data
    """

    def __init__(self, config):
        loader = Loader(config)
        super(RetrievalLoader, self).__init__(
            loader.descriptions, loader.shapes)

        self.txt_vectorization = loader.txt_vectorization
        self.length_voc = len(self.txt_vectorization.voc_list)


def parse_directory_for_nrrd(path):
    shapes = dict()
    shapes['modelId'] = []
    shapes['data'] = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".nrrd"):
                train_data, _ = nrrd.read(
                    os.path.join(root, file), index_order='C')
                shapes['modelId'].append(file.replace('.nrrd', ''))
                shapes['data'].append(train_data)

    return shapes
