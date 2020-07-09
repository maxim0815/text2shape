import torch

def TripletLoss(batch):
    """
    :param batch: type = list, batch of random triplets
    :return: mean loss over batch size
    TODO: Parallelise using batch with type tensor.
    """
    margin = 0.7
    p_norm = 2
    loss_sum = 0
    for i in range(len(batch)):
        a, p, n = batch[i]
        loss = torch.max(torch.norm(a - p, p=p_norm) - torch.norm(a - n, p=p_norm) + margin, 0)
        loss_sum += loss[0]
    mean_loss = loss_sum / len(batch)
    return mean_loss
