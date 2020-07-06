import argparse

from NatLangPreprocessor import NatLangPreprocessor


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str,
                        help="caption that needs to be preprocessed")
    parser.add_argument('output_csv', type=str,
                        help="preprocessed output file")
    parser.add_argument('output_voc_csv', type=str,
                        help="file where vocabular is saved")
    args = parser.parse_args()
    return args

def main(args):
    # run for preprocessor
    prep = NatLangPreprocessor(args.input_csv)
    prep.preprocess()

    prep.save_vocabulary(args.output_voc_csv)
    prep.save_data(args.output_csv)


if __name__ == '__main__':
    args =parse_arguments()
    main(args)
