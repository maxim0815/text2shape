import torch


def triplet_loss(a, p, n, p_norm=2, margin=0.7):

    """
    :param batch: type = list, batch of random triplets
    :return: mean loss over batch size, norm values
    TODO: Parallelise using batch with type tensor.
    """

    norm_p = torch.norm(a - p, p=p_norm, dim=1, keepdim=True)
    norm_n = torch.norm(a - n, p=p_norm, dim=1, keepdim=True)
    loss = torch.max(torch.cat([(norm_p - norm_n + margin), torch.zeros_like(norm_p)], dim=1), dim=1)
    mean_loss = torch.mean(loss[0])
    return mean_loss, norm_p, norm_n


if __name__ == "__main__":
    a, p, n = torch.randn((32, 128)), torch.randn((32, 128)), torch.randn((32, 128))
    print(triplet_loss(a, p, n), triplet_loss(a, p, n)[2].shape)