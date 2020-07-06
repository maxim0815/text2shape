import numpy as np
import pandas as pd
import spacy
import language_tool_python


class NatLangPreprocessor():
    def __init__(self, csv_dir):

        try:
            self.csv_data = pd.read_csv(csv_dir)
        except:
            print("error reading csv file")

        self.prep_data = dict()
        self.unique_tokens = dict()
        for key in self.csv_data.keys():
            self.prep_data.update({key : []})
        self.tool = language_tool_python.LanguageTool('en-US')
    
    def _preprocess(self, max_length=96):
        nlp = spacy.load("en_core_web_sm")
        for i, description in enumerate(self.csv_data['description']):
            description_check_locs = self.tool.correct(description)
            doc = nlp(description_check_locs)
            preprocessed_description = []
            for token in doc:
                preprocessed_description.append(token.norm_) # TODO might be lemma_ however returns PRON sometimes i.e. for it
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
    
    def save_vocabulary(self, dir_name = '../data/vocabulary.csv'):
        vocabulary = []
        for key in self.unique_tokens.keys():
            if self.unique_tokens[key] > 2:
                vocabulary.append(key)
        vocabulary.append("UNK")
  
        # Calling DataFrame constructor on list 
        df = pd.DataFrame(vocabulary, columns =['Vocabulary'])
        df.to_csv(dir_name, index=False)


    def save_data(self, dir_name = '../data/preprocessed_captions.csv'):
        # Calling DataFrame constructor on dict
        df = pd.DataFrame.from_dict(self.prep_data)
        df.to_csv(dir_name, index=False)


class NatLangLoader():
    def __init__(self, proc_desc_dir = '../data/preprocessed_captions.csv', voc_dir = '../data/vocabulary.csv'):
        try:
            self.data = pd.read_csv(proc_desc_dir)
            self.vocabulary = pd.read_csv(voc_dir)
        except:
            print("error reading csv file")

    def get_vocabulary(self):
        return self.vocabulary['Vocabulary']

    def get_encoded_desc(self, id):
        # print(self.data['id'])
        # idx = self.data['id'].index(id)
        # description = self.data['description'][idx]
        # encoded = np.zeros(len(description))
        # for i, token in enumerate(description):
        #     encoded_desc[i] = self.vocabulary.index(token)
        # return encoded_desc
        return

if __name__ == '__main__':
    preprocessor = True
    if preprocessor == True:
        # run for preprocessor
        prep = NatLangPreprocessor('../data/captions.tablechair.csv')
        prep._preprocess()
        prep.save_vocabulary()
        prep.save_data()

    # run for loading preprocessed data
    lang = NatLangLoader()
    print(lang.get_vocabulary())
    print(lang.get_encoded_desc(118458))
