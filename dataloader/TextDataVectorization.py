import sys
import pandas as pd
import numpy as np


class TxtVectorization(object):
    '''
    uses vocabulary to tranform sentences to vector or vice versa
    '''

    def __init__(self, voc_dir):
        try:
            csv_voc = pd.read_csv(voc_dir).to_dict()
            voc_dict = csv_voc["vocabulary"]
            self.voc_list = []
            for key, value in voc_dict.items():
                self.voc_list.append(value)

        except:
            sys.exit("ERROR! TxtVectorization is not able to read csv file")

    def description2vector(self, description):
        words = description.split(" ")
        vector = np.zeros((len(self.voc_list), 1))
        for word in words:
            # UNK
            if word not in self.voc_list:
                vector[-1] = 1
            else:
                index = self.voc_list.index(word)
                vector[index] = 1
        return vector

    def vector2description(self, vector):
        description = ""
        for i in range(len(vector)):
            # UNK reached
            if i == len(vector)-1:
                break
            if vector[i] == 1:
                description += self.voc_list[i] + " "
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
