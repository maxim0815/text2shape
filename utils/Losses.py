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


def lossless_triplet_loss(a, p, n, N = 128, beta=128, eps=1e-8):

    # pos = torch.square(a-p)
    # neg = torch.square(a-n)
    pos = torch.norm(a - p, p=2, dim=1, keepdim=True)
    neg = torch.norm(a - n, p=2, dim=1, keepdim=True)

    pos_dist = -torch.log(- pos/beta + 1 + eps)
    neg_dist = -torch.log(- (N  - neg)/beta + 1 + eps)
    
    # compute loss
    loss = neg_dist + pos_dist
    return torch.sum(loss), pos_dist, neg_dist


def multi_class_npair_loss(a, p, n):
    """
    a and p are single samples, while n is a tensor of multiple negative samples
    """
    ap = torch.dot(a, p).exp()
    an = (n @ a).exp().sum()
    return -torch.log(ap/(ap+an))


def nt_xent_loss(a, p, n, tau=2.0):
    """
    a and p are single samples, while n is a tensor of multiple negative samples
    """
    ap = (torch.dot(a, p) / tau).exp()
    an = ((n @ a) / tau).exp().sum()
    return -torch.log(ap / an)


if __name__ == "__main__":
    a, p, n = torch.randn((32, 128)), torch.randn((32, 128)), torch.randn((32, 128))
    # print(triplet_loss(a, p, n), triplet_loss(a, p, n)[2].shape)
    # print(multi_class_npair_loss(a[0], p[0], n))
    # print(nt_xent_loss(a[0], p[0], n))
