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