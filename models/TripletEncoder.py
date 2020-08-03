import torch
import os
import sys

from models.Networks import ShapeEncoder, TextEncoder
from dataloader.DataLoader import TripletText2Shape, TripletShape2Text
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

    def update(self, batch, batch_2=0):
        '''
        how does the batch looks like???
        so far:
            batch is list of class Triplet
        batch_2 != 0:
            one batch t2s and another batch s2t and same time
        '''
        self.shape_encoder.train()
        self.text_encoder.train()
        if batch_2 == 0:
            anchor, pos, neg = self.__forward_batch(batch)
            # calculate triplet loss
            loss, dist_pos, dist_neg = triplet_loss(anchor, pos, neg)
        else:
            anchor_1, pos_1, neg_1 = self.__forward_batch(batch)
            anchor_2, pos_2, neg_2 = self.__forward_batch(batch_2)

            loss_1, dist_pos_1, dist_neg_1 = triplet_loss(anchor_1, pos_1, neg_1)
            loss_2, dist_pos_2, dist_neg_2 = triplet_loss(anchor_2, pos_2, neg_2)

            loss = loss_1 + loss_2
            dist_pos = dist_pos_1 + dist_pos_2
            dist_neg = dist_neg_1 + dist_neg_2


        # accuray
        pred = (dist_pos - dist_neg).cpu().data
        acc = (pred > 0).sum()*1.0/dist_pos.size()[0]

        eval_dict = {"loss": loss.item(), "accuracy": acc}

        self.optimizer_shape.zero_grad()
        self.optimizer_text.zero_grad()

        # backprop
        loss.backward()

        self.optimizer_shape.step()
        self.optimizer_text.step()

        return eval_dict

    def predict(self, batch, batch_2=0):

        self.shape_encoder.eval()
        self.text_encoder.eval()

        if batch_2 == 0:
            anchor, pos, neg = self.__forward_batch(batch)
            # calculate triplet loss
            loss, dist_pos, dist_neg = triplet_loss(anchor, pos, neg)
        else:
            anchor_1, pos_1, neg_1 = self.__forward_batch(batch)
            anchor_2, pos_2, neg_2 = self.__forward_batch(batch_2)

            loss_1, dist_pos_1, dist_neg_1 = triplet_loss(anchor_1, pos_1, neg_1)
            loss_2, dist_pos_2, dist_neg_2 = triplet_loss(anchor_2, pos_2, neg_2)

            loss = loss_1 + loss_2
            dist_pos = dist_pos_1 + dist_pos_2
            dist_neg = dist_neg_1 + dist_neg_2

        pred = (dist_pos - dist_neg).cpu().data
        acc = (pred > 0).sum()*1.0/dist_pos.size()[0]

        eval_dict = {"loss": loss.item(), "accuracy": acc}

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

    def __forward_batch(self, batch):
        if isinstance(batch[0], TripletShape2Text):
            shape_batch, pos_desc_batch, neg_desc_batch = self.triplet_list_to_tensor(
                batch)
            # set requires_grad to true
            shape_batch.requires_grad_()
            pos = self.text_encoder(pos_desc_batch)
            neg = self.text_encoder(neg_desc_batch)
            anchor = self.shape_encoder(shape_batch)

        if isinstance(batch[0], TripletText2Shape):
            desc_batch, pos_shape_batch, neg_shape_batch = self.triplet_list_to_tensor(
                batch)
            pos_shape_batch.requires_grad_()
            neg_shape_batch.requires_grad_()
            pos = self.shape_encoder(pos_shape_batch)
            neg = self.shape_encoder(neg_shape_batch)
            anchor = self.text_encoder(desc_batch)
        return anchor, pos, neg

    def triplet_list_to_tensor(self, batch):
        '''
        triplet list get seperatet into tree tensors
        '''
        if isinstance(batch[0], TripletShape2Text):
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
        
        if isinstance(batch[0], TripletText2Shape):
            bs = len(batch)
            max_length = 96
            pos_shape_batch = torch.zeros((bs, 32, 32, 32, 4))
            neg_shape_batch = torch.zeros((bs, 32, 32, 32, 4))
            desc_batch = torch.zeros((bs, max_length)).long()
            for i, triplet in enumerate(batch):
                pos_shape_batch[i] = torch.from_numpy(triplet.pos_shape)
                neg_shape_batch[i] = torch.from_numpy(triplet.neg_shape)
                desc_batch[i] = torch.from_numpy(triplet.desc)
            
            pos_shape_batch = pos_shape_batch.to(self.device)
            neg_shape_batch = neg_shape_batch.to(self.device)
            desc_batch = desc_batch.to(self.device)

            return desc_batch, pos_shape_batch, neg_shape_batch