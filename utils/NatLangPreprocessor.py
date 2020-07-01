import numpy as np
import pandas as pd
import spacy
import language_tool_python


class NatLangPreprocessor():
    def __init__(self, csv_dir):
        self.tool = language_tool_python.LanguageTool('en-US')
        try:
            self.csv_data = pd.read_csv(csv_dir)
            self.prep_data = dict()
            self.unique_tokens = dict()
            for key in self.csv_data.keys():
                self.prep_data.update({key : []})
        except:
            print("error reading csv file")
    
    def _preprocessing(self, max_length=96):
        nlp = spacy.load("en_core_web_sm")
        for i, description in enumerate(self.csv_data['description']):
            description_check_locs = self.tool.correct(description)
            doc = nlp(description_check_locs)
            preprocessed_description = []
            for token in doc:
                preprocessed_description.append(token.norm_) # TODO should be lemma_ however returns PRON sometimes i.e. for it
            self._count(preprocessed_description)
            if len(preprocessed_description) < max_length:
                for key in self.csv_data.keys():
                    if key == 'description':
                        self.prep_data['description'].append(preprocessed_description)
                    else:
                        self.prep_data[key].append(self.csv_data[key][i])

    def _count(self, description):
        # check it token is in dictionary, either increment or add to dict
        for token in description:
            if token in self.unique_tokens:
                self.unique_tokens.update({token : self.unique_tokens[token]+1})
            else:
                self.unique_tokens.update({token : 1})

    def _rm_low_freqency(self):
        to_remove = []
        for key in self.unique_tokens.keys():
            if self.unique_tokens[key] < 3:
                to_remove.append(key)

        for description in self.prep_data['description']:
            for i, token in enumerate(description):
                if token in to_remove:
                    description[i] = "UNK"

                




if __name__ == '__main__':
    prep = NatLangPreprocessor('../data/captions.tablechair.csv')
    prep._preprocessing()
    prep._rm_low_freqency()
    print(prep.prep_data)
