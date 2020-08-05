import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store', type=str,
                        nargs='+', help="directory to .csv files for plot")
    args = parser.parse_args()
    return args

def file_to_lists(data):
    """
    pandas stores dict within dict which contains idx as key
    this functions converts these dicts into one list
    """

    for key, _ in data.items():
        data_list = []
        for _, v in data[key].items():
            data_list.append(v)
        data[key] = data_list

def main(args):
    """
    Do not forget to put the right Axislabel into the file
    """

    for file in args.input:
        name = file.split("/")[-1].replace(".csv", "")
        data = pd.read_csv(file).to_dict()
        file_to_lists(data)
        plt.plot(data['Step'], data['Value'], label=name)

        print("HUI")

    plt.xlabel("Epoch")
    plt.ylabel("NDCG")
    plt.title("S2T NDCG score")
    plt.legend()
    plt.savefig("output.png", dpi=500)
    plt.close()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)