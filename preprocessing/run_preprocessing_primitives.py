import argparse

from NatLangPreprocessor import NatLangPreprocessor
# needed for import from starting directory
import sys
import os
sys.path.append(os.getcwd())
print(sys.path)

from dataloader.DataLoader import parse_primitives

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str,
                        help="direcroty of all folders")
    parser.add_argument('categorize', type=str,
                        help="shape or shape_color")
    parser.add_argument('output_csv', type=str,
                        help="preprocessed output file")
    parser.add_argument('output_voc_csv', type=str,
                        help="file where vocabular is saved")
    args = parser.parse_args()
    return args

def main(args):
    # run for preprocessor
    _, descriptions = parse_primitives(args.directory, args.categorize)
    prep = NatLangPreprocessor(descriptions)
    prep.preprocess()

    prep.save_vocabulary(args.output_voc_csv)
    prep.save_data(args.output_csv)


if __name__ == '__main__':
    args =parse_arguments()
    main(args)
