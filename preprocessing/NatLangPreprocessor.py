import numpy as np
import pandas as pd
import spacy
import language_tool_python
import sys


class NatLangPreprocessor():
    '''
    - parse through all descriptions and removes errors
    - saves vorabulary with all words which appear more than twice
    - saves preprocessed data with form "the table is red . the table is round"
      each word within description can be saved to list with space as delimitter
    '''
    def __init__(self, csv_dir):

        try:
            self.csv_data = pd.read_csv(csv_dir)
        except:
            sys.exit("ERROR! Preprocessor is not able to read csv file")

        self.prep_data = dict()
        self.unique_tokens = dict()
        for key in self.csv_data.keys():
            self.prep_data.update({key: []})
        self.tool = language_tool_python.LanguageTool('en-US')

    def preprocess(self, max_length=96):
        nlp = spacy.load("en_core_web_sm")
        length = len(self.csv_data['description'])
        for i, description in enumerate(self.csv_data['description']):
            if type(description) == str:
                description_check_locs = self.tool.correct(description)
                doc = nlp(description_check_locs)
                preprocessed_description = ""
                word_list = []
                for token in doc:
                    # TODO might be lemma_ however returns PRON sometimes i.e. for it
                    preprocessed_description += token.norm_ + " "
                    word_list.append(token.norm_)
                # remove last space
                preprocessed_description = preprocessed_description[:-1]
                self._count(word_list)
                if len(word_list) < max_length:
                    for key in self.csv_data.keys():
                        if key == 'description':
                            self.prep_data['description'].append(
                                preprocessed_description)
                        else:
                            self.prep_data[key].append(self.csv_data[key][i])
            print('Preprocessing text {:.2f} %'.format(i/length*100), end='\r')

    def _count(self, description):
        # check it token is in dictionary, either increment or add to dict
        for token in description:
            if token in self.unique_tokens:
                self.unique_tokens.update({token: self.unique_tokens[token]+1})
            else:
                self.unique_tokens.update({token: 1})

    def save_vocabulary(self, dir_name):
        vocabulary = []
        for key in self.unique_tokens.keys():
            if self.unique_tokens[key] > 2:
                vocabulary.append(key)
        vocabulary.append("UNK")
        print("Saved vocabulary: " + dir_name)

        # Calling DataFrame constructor on list
        df = pd.DataFrame(vocabulary, columns=['vocabulary'])
        df.to_csv(dir_name, index=False)

    def save_data(self, dir_name):
        # Calling DataFrame constructor on dict
        df = pd.DataFrame.from_dict(self.prep_data)
        df.to_csv(dir_name, index=False)
        print("Saved preprocessed data: " + dir_name)