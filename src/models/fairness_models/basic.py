# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn
import scipy as sp

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class BASIC(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(BASIC, self).__init__(config, dataloader)

        self.base_model = config['recommendation_model']
        self.neg_slope = config['neg_slope']
        self.num_features = len(config['feature_columns'])

        

        pretrained_user_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_user_representation.npy'), allow_pickle=True)
        self.user_representation = torch.FloatTensor(pretrained_user_representation).to(self.device)
        # self.user_representation.requires_grad = True 
        pretrained_item_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_item_representation.npy'), allow_pickle=True)
        self.item_representation = torch.FloatTensor(pretrained_item_representation).to(self.device)
        # self.item_representation.requires_grad = True 

        self.interaction_matrix = dataloader.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).float().to_dense()
        row_sums = self.interaction_matrix.sum(axis=-1)
        self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
        self.interaction_matrix = self.interaction_matrix.to(self.device)






    def get_user_embedding(self, users):
        return self.user_representation[users]

    def get_user_aggregated_item_embedding(self, users):
        user_aggregated_item_embeddings = torch.mm(self.interaction_matrix[users], self.item_representation)
        return user_aggregated_item_embeddings

    def save_pretrained_representation(self):
        pass

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_fair_representation(self):
        pass