
from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.data import get_dataset_statistics




class LYContext(nn.Module):
    def __init__(self, config):
        super(LYContext, self).__init__()
        self.cfg = config.clone() 
        self.statistics = get_dataset_statistics(self.cfg)
        self.obj_classes, self.rel_classes, self.att_classes = self.statistics['obj_classes'], self.statistics['rel_classes'], self.statistics['att_classes']
        self.union = len(self.obj_classes) * len(self.obj_classes)
        self.register_buffer("entity_list", torch.zeros(len(self.obj_classes), len(self.obj_classes)).fill_diagonal_(1))
        self.register_buffer("relation_list", torch.zeros(self.union, self.union).fill_diagonal_(1))
        self.spa_head =nn.Sequential(*[
            make_fc(25, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            make_fc(128, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            make_fc(512, 1024),
            nn.ReLU(inplace=True)])
        '''self.attention = nn.Sequential(*[
            make_fc(1024, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            make_fc(256, 1),
            nn.ReLU(inplace=True)])
        
        self.label_union =nn.Sequential(*[
            make_fc(len(self.obj_classes)*len(self.obj_classes), 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            make_fc(256, 1024),
            nn.ReLU(inplace=True)])'''
        self.subject =nn.Sequential(*[
            make_fc(len(self.obj_classes), 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            make_fc(256, 1024),
            nn.ReLU(inplace=True)])
        self.object =nn.Sequential(*[
            make_fc(len(self.obj_classes), 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            make_fc(256, 1024),
            nn.ReLU(inplace=True)])
        #self.weight0 = torch.autograd.Variable(torch.ones(1).to(self.device), requires_grad=True)
        self.weight0 = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.weight1 = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
        #self.weight3 = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)

        self.all_product = nn.Sequential(*[
            make_fc(1024, 2048),
            nn.BatchNorm1d(2048),
            make_fc(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True)],)

    def calcul_mask(self, mask_l, mask_r, obj_l, obj_r):
        mask_left = mask_l>(obj_l.unsqueeze(1))
        mask_right = mask_r<(obj_r.unsqueeze(1))
        mask = mask_left*mask_right
        return mask

    def make_mask(self, sub_1, sub_2):
        ave = ((sub_2 - sub_1)/3).unsqueeze(1)
        start = sub_1.unsqueeze(1)
        mask_l = torch.cat([start, start+ave, start+2*ave, start+3*ave, start+10000], dim=1)
        mask_r = torch.cat([start-10000, start, start+ave, start+2*ave, start+3*ave], dim=1)
        return mask_l, mask_r

    def spatical(self, input_1, input_2):
        mask_x_l, mask_x_r = self.make_mask(input_1[:, 0], input_1[:, 2])
        mask_y_l, mask_y_r = self.make_mask(input_1[:, 1], input_1[:, 3])
        mask_x = self.calcul_mask(mask_x_l, mask_x_r, input_2[:,0], input_2[:, 2])
        mask_y = self.calcul_mask(mask_x_l, mask_x_r, input_2[:,1], input_2[:, 3])
        mask = mask_x.unsqueeze(1) * mask_y.unsqueeze(-1)
        mask = torch.reshape(mask, (len(mask), 25))
        return mask

    def embedding(self, indx, length):
        new_new = [0 for i in range(length)]
        new_new[indx] = 1
        return new_new
    
    def obj_embedding(self, indx, length):
        new_new = [-1000 for i in range(length)]
        new_new[indx] = 1000
        return new_new

    # def forward(self, roi_features, proposals, rel_pair_idxs, rel_labels, rel_binarys,  union_features, inter_features, logger, ctx_average=False):
    def forward(self, proposals, obj_dict_list, rel_pair_idxs):
        subject = []
        obbject = []
        union = []
        subject_box = []
        object_box = []
        for i in range(len(proposals)):
            #obj_pred_label = torch.max(proposals[i].get_field("predict_logits"), 1).indices
            obj_pred_label = torch.max(obj_dict_list[i], 1).indices

            bbox_wait = proposals[i].bbox
            subject_box.append(bbox_wait[rel_pair_idxs[i][:, 0]].long())
            object_box.append(bbox_wait[rel_pair_idxs[i][:, 1]].long())

            s = obj_pred_label[rel_pair_idxs[i][:, 0].long()]
            o = obj_pred_label[rel_pair_idxs[i][:, 1].long()]

            subject.append(self.entity_list[s.long()])
            obbject.append(self.entity_list[o.long()])
            #union.append(self.relation_list[(s*len(self.obj_classes)+o).long()])

        subject_box = torch.cat(subject_box, dim=0)
        object_box = torch.cat(object_box, dim=0)

        
        spatical_Fs = self.spatical(subject_box, object_box)
        #label_unions = torch.cat(union, dim=0)
        subjects = torch.cat(subject, dim=0)
        objects = torch.cat(obbject, dim=0)
        
        spatical_feature = self.spa_head(spatical_Fs.float())
        #spa_attention = self.attention(spatical_feature)
        #label_union_feature = self.label_union(label_unions)
        subject_feature = self.subject(subjects)
        object_feature = self.subject(objects)
        all_feature = spatical_feature*self.weight0 + subject_feature*self.weight1 + object_feature*self.weight2
        
        rel_dicts = self.all_product(all_feature)
        return rel_dicts