import torch
import numpy as np


def find_nn_text_2_text(model, input_, loader, k):
    model.eval()
    input_ = torch.from_numpy(input_).long()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = model(input_)
    loss_L2 = np.array([])
    for i in range(loader.get_description_length()):
        print("Calculate L2 loss for {} of {}".format(
            i, loader.get_description_length()), end='\r')
        data = loader.get_description(i)
        data = torch.from_numpy(data).long()
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


def find_nn_shape_2_shape(model, input_, loader, k):
    model.eval()

    input_ = torch.from_numpy(input_).float()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = model(input_)
    loss_L2 = np.array([])
    for i in range(loader.get_shape_length()):
        print("Calculate L2 loss for {} of {}".format(
            i, loader.get_shape_length()), end='\r')
        data = loader.get_shape(i)
        data = torch.from_numpy(data).float()
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


def find_nn_shape_2_text(shape_model, text_model, input_, loader, k):
    shape_model.eval()
    text_model.eval()

    input_ = torch.from_numpy(input_).float()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = shape_model(input_)
    loss_L2 = np.array([])
    for i in range(loader.get_description_length()):
        print("Calculate L2 loss for {} of {}".format(
            i, loader.get_description_length()), end='\r')
        data = loader.get_description(i)
        data = torch.from_numpy(data).long()
        if torch.cuda.is_available():
            data = data.to('cuda')
        output = text_model(data)
        loss_L2 = np.append(loss_L2, criterion(output_q, output).item())
    print()

    idx = np.argsort(loss_L2)
    # start with 1 to remove comparison with its own
    closest_idx = idx[1:k+1]
    closest_dist = loss_L2[idx[1:k+1]]
    return closest_idx, closest_dist


def find_nn_text_2_shape(text_model, shape_model,  input_, loader, k):
    shape_model.eval()
    text_model.eval()

    input_ = torch.from_numpy(input_).long()
    if torch.cuda.is_available():
        input_ = input_.to('cuda')

    criterion = torch.nn.MSELoss()

    output_q = text_model(input_)
    loss_L2 = np.array([])
    for i in range(loader.get_shape_length()):
        print("Calculate L2 loss for {} of {}".format(
            i, loader.get_shape_length()), end='\r')
        data = loader.get_shape(i)
        data = torch.from_numpy(data).float()
        if torch.cuda.is_available():
            data = data.to('cuda')
        output = shape_model(data)
        loss_L2 = np.append(loss_L2, criterion(output_q, output).item())
    print()

    idx = np.argsort(loss_L2)
    # start with 1 to remove comparison with its own
    closest_idx = idx[1:k+1]
    closest_dist = loss_L2[idx[1:k+1]]
    return closest_idx, closest_dist


def calculate_ndcg(idx_neighbor, idx_input, dataloader, n_neighbors, metric):
    """
    idx_neighbor:       indicies of all nearest  neighbors
    idx_input:          index of what we computed the nearest neighbors to
    dataloade:          backend holding all data
    n_neighbors:        number of calculated neighbors == len(idx_neighbor)
    metric:             t2t - t2s - s2t - s2s
    """

    rel_score = np.zeros((n_neighbors))
    # ideal would be all nearest neighbor are from same category as input
    rel_score_ideal = np.ones((n_neighbors))

    if metric == "t2t":
        true_label = dataloader.descriptions['category'][idx_input]
        for i, id in enumerate(idx_neighbor):
            label = dataloader.descriptions['category'][id]
            rel_score[i] = np.asarray(label == true_label)
    
    if metric == "t2s":
        true_label = dataloader.descriptions['category'][idx_input]
        for i, id in enumerate(idx_neighbor):
            label = dataloader.shapes['category'][id]
            rel_score[i] = np.asarray(label == true_label)

    if metric == "s2t":
        true_label = dataloader.shapes['category'][idx_input]
        for i, id in enumerate(idx_neighbor):
            label = dataloader.descriptions['category'][id]
            rel_score[i] = np.asarray(label == true_label)
    
    if metric == "s2s":
        true_label = dataloader.shapes['category'][idx_input]
        for i, id in enumerate(idx_neighbor):
            label = dataloader.shapes['category'][id]
            rel_score[i] = np.asarray(label == true_label)
        

    # Compute Discounted Cumulative Gain
    nominator = np.exp2(rel_score) - 1
    denominator = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.sum(nominator/denominator)
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.sum(dcg_n_ideal/denominator)

    # Compute normalized dcg
    ndcg = dcg / dcg_ideal

    return ndcg
