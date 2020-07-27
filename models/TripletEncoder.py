import torch
import os
import sys

from models.Networks import ShapeEncoder, TextEncoder

from utils.Losses import triplet_loss


class TripletEncoder(object):
    def __init__(self, config, voc_size):
        self.shape_encoder = ShapeEncoder()
        # TODO: voc_size to config??
        self.text_encoder = TextEncoder(voc_size)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.shape_encoder = self.shape_encoder.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        lr = config['hyper_parameters']["lr"]
        mom = config['hyper_parameters']["mom"]

        self.save_directory = config['directories']['model_save']
        self.load_directory = []
        self.load_directory.append(config['directories']['shape_model_load'])
        self.load_directory.append(config['directories']['text_model_load'])

        # TODO: maybe each optimizer gets its own params?
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
        shape_batch, pos_desc_batch, neg_desc_batch = self.triplet_list_to_tensor(
            batch)

        self.shape_encoder.train()
        self.text_encoder.train()

        # set requires_grad to true
        shape_batch.requires_grad_()

        pos_out = self.text_encoder(pos_desc_batch)
        neg_out = self.text_encoder(neg_desc_batch)
        shape_out = self.shape_encoder(shape_batch)

        # calculate triplet loss

        loss, dist_pos, dist_neg = triplet_loss(shape_out, pos_out, neg_out)

        #accuray
        pred = (dist_pos - dist_neg).cpu().data
        acc = (pred > 0).sum()*1.0/dist_pos.size()[0] 

        eval_dict = {"loss" : loss.item(), "accuracy" : acc}

        self.optimizer_shape.zero_grad()
        self.optimizer_text.zero_grad()

        # backprop
        loss.backward()

        self.optimizer_shape.step()
        self.optimizer_text.step()

        return eval_dict

    def predict(self, batch):

        shape_batch, pos_desc_batch, neg_desc_batch = self.triplet_list_to_tensor(
            batch)

        self.shape_encoder.eval()
        self.text_encoder.eval()

        pos_out = self.text_encoder(pos_desc_batch)
        neg_out = self.text_encoder(neg_desc_batch)
        shape_out = self.shape_encoder(shape_batch)

        # calculate triplet loss
        loss, dist_pos, dist_neg = triplet_loss(shape_out, pos_out, neg_out)

        pred = (dist_pos - dist_neg).cpu().data
        acc = (pred > 0).sum()*1.0/dist_pos.size()[0] 

        eval_dict = {"loss" : loss.item(), "accuracy" : acc}

        return eval_dict

    def save_models(self):
        name = "shape_encoder.pt"
        file_name = os.path.join(self.save_directory, name)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        torch.save(self.shape_encoder.state_dict(), file_name)
        name = "text_encoder.pt"
        file_name = os.path.join(self.save_directory, name)
        torch.save(self.text_encoder.state_dict(), file_name)

    def load_models(self):
        try:
            # load shape net
            temp_net = torch.load(
                self.load_directory[0], map_location=self.device)
            self.shape_encoder = self.shape_encoder.to(self.device)
            self.shape_encoder.load_state_dict(temp_net)
            # load text net
            temp_net = torch.load(
                self.load_directory[1], map_location=self.device)
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.load_state_dict(temp_net)
        except:
            sys.exit("ERROR! Failed loading models into TripletEncoder")

    def triplet_list_to_tensor(self, batch):
        '''
        triplet list get seperatet into tree tensors
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

        shape_batch = shape_batch.to(self.device)
        pos_desc_batch = pos_desc_batch.to(self.device)
        neg_desc_batch = neg_desc_batch.to(self.device)

        return shape_batch, pos_desc_batch, neg_desc_batch
