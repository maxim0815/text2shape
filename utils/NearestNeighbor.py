import torch
import numpy as np


def find_nn(model, input_, data_list, k):
    """
    Find the k nearest neighbors (NNs) of given input, in the feature space of the specified mode.
    Args:
        model: the model for computing the features
        input: the input (shape, text) of which to find the NNs
        data_list: the loader for the dataset in which to look for the NNs
        k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    model.eval()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = model(input_)
    loss_L2 = np.array([])
    for i, data in enumerate(data_list):
        print("Calculate L2 loss for {} of {}".format(
            i, len(data_list)), end='\r')
        if torch.cuda.is_available():
            data = data.to('cuda')
        output = model(data)
        loss_L2 = np.append(loss_L2, criterion(output_q, output).item())
    print()

    idx = np.argsort(loss_L2)
    # start with 1 to remove comparison with its own
    closest_idx = idx[1:k+1]
    closest_dist = loss_L2[idx[1:k+1]]
    return closest_idx, closest_dist


def find_nn_cross_modal(model1, model2, input_, data_list, k):
    """
    Find the k nearest neighbors (NNs) of given input, in the feature space of the specified mode.
    Args:
        model1:    model for calculating input_
        model2:    model for computing the features of data_list
        input:     the input (shape, text) of which to find the NNs
        data_list: the loader for the dataset in which to look for the NNs
        k:         the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    model1.eval()
    model2.eval()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = model1(input_)
    loss_L2 = np.array([])
    for i, data in enumerate(data_list):
        print("Calculate L2 loss for {} of {}".format(
            i, len(data_list)), end='\r')
        if torch.cuda.is_available():
            data = data.to('cuda')
        output = model2(data)
        loss_L2 = np.append(loss_L2, criterion(output_q, output).item())
    print()

    idx = np.argsort(loss_L2)
    # start with 1 to remove comparison with its own
    closest_idx = idx[1:k+1]
    closest_dist = loss_L2[idx[1:k+1]]
    return closest_idx, closest_dist