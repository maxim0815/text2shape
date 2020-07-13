import torch

def triplet_loss(a, p, n):

    """
    :param batch: type = list, batch of random triplets
    :return: mean loss over batch size, norm values
    TODO: Parallelise using batch with type tensor.
    """
    margin = 0.7
    p_norm = 2
    loss_sum = 0
    norm_p = []
    norm_n = []
    for i in range(len(a)):
        norm_p.append(torch.norm(a - p, p=p_norm))
        norm_n.append(torch.norm(a - n, p=p_norm))
        loss = torch.max(norm_p[i] - norm_n[i] + margin, 0)
        loss_sum += loss[0]
    mean_loss = loss_sum / len(a)
    return mean_loss, norm_p, norm_n
