import sys
import pandas as pd
import numpy as np


class TxtVectorization(object):
    '''
    uses vocabulary to tranform sentences to vector or vice versa
    '''

    def __init__(self, voc_dir, max_desc_length=96):
        try:
            csv_voc = pd.read_csv(voc_dir).to_dict()
            voc_dict = csv_voc["vocabulary"]
            self.voc_list = []
            self.max_desc_length = max_desc_length
            for key, value in voc_dict.items():
                self.voc_list.append(value)

        except:
            sys.exit("ERROR! TxtVectorization is not able to read csv file")

    def description2vector(self, description):
        words = description.split(" ")
        vector = np.zeros(self.max_desc_length, dtype=np.int32)
        for i, word in enumerate(words):
            if word in self.voc_list:
                vector[i] = self.voc_list.index(word)
            else:
                vector[i] = self.voc_list.index("UNK")
        return vector

    def vector2description(self, vector):
        description = ""
        for i in range(len(vector)):
            word = self.voc_list[vector[i]]
            if word == "END":
                break
            description += word + " "
        # remove last space
        description = description[:-1]
        return description


#test = TxtVectorization("data/voc.csv")
#pre = pd.read_csv("data/preprocessed.captions.csv").to_dict()
#description = pre["description"][0]
#
#vector = test.description2vector(description)
#
#desc = test.vector2description(vector)
#print("HUI")
