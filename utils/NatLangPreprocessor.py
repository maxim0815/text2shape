import numpy as np
import csv


class NatLangPreprocessor():
    def __init__(self, csv_dir):
        with open(csv_dir) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(row)
                    line_count += 1
                else:
                    





        id,modelId,description,category,topLevelSynsetId,subSynsetId