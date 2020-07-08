import torch

from models.Networks import ShapeEncoder, TextEncoder


class TripletEncoder(object):
    def __init__(self, config, voc_size):
        self.shape_encoder = ShapeEncoder()
        #TODO: voc_size to config??
        self.text_encoder = TextEncoder(voc_size)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.shape_encoder = self.shape_encoder.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        lr = config['hyper_parameters']["lr"]
        mom = config['hyper_parameters']["mom"]

        #TODO: maybe each optimizer gets its own params?
        self.optimizer_shape = torch.optim.SGD(
            self.shape_encoder.parameters(), lr=lr, momentum=mom)
        self.optimizer_text = torch.optim.SGD(
            self.text_encoder.parameters(), lr=lr, momentum=mom)

    def update(self, batch):
        '''
        how does the batch looks like???
        so far:
            batch is list of class Triplet
        '''
        bs = len(batch)
        max_length = 96
        pos_desc_batch = torch.zeros((bs, max_length)).long()
        neg_desc_batch = torch.zeros((bs, max_length)).long()
        shape_batch = torch.zeros((bs, 32, 32, 32, 4))
        for i, triplet in enumerate(batch):
            pos_desc_batch[i] = torch.from_numpy(triplet.pos_desc).long()
            neg_desc_batch[i] = torch.from_numpy(triplet.neg_desc).long()
            shape_batch[i] = torch.from_numpy(triplet.shape)
        
        if torch.cuda.is_available():
            pos_desc_batch.cuda()
            neg_desc_batch.cuda()
            shape_batch.cuda()

        self.shape_encoder.train()
        self.text_encoder.train()
        
        # set requires_grad to true
        shape_batch.requires_grad_()

        pos_out = self.text_encoder(pos_desc_batch)
        neg_out = self.text_encoder(neg_desc_batch)
        shape_out = self.shape_encoder(shape_batch)

        # calculate triplet loss
        eval_dict = {"loss" : 1, "accuracy" : 0.9}

        self.optimizer_shape.zero_grad()
        self.optimizer_text.zero_grad()
        
        #backprop

        self.optimizer_shape.step()
        self.optimizer_text.step()

        return eval_dict

