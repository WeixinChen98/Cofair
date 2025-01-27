# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
LATTICE
################################################
Reference:
    https://github.com/CRIPAC-DIG/LATTICE
    ACM MM'2021: [Mining Latent Structures for Multimedia Recommendation] 
    https://arxiv.org/abs/2104.09036
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class LightGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.weight_size = config['weight_size']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix_ = dataset.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix_).float().to_dense()
        row_sums = self.interaction_matrix.sum(axis=-1)
        self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
        self.interaction_matrix = self.interaction_matrix.to(self.device)
        self.norm_adj = self.get_adj_mat()
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.n_ui_layers = len(self.weight_size)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def pre_epoch_processing(self):
        return 

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix_.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            #print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        self.pretrained_user_representation, self.pretrained_item_representation = self.forward(self.norm_adj)

        u_g_embeddings = self.pretrained_user_representation[users]
        pos_i_g_embeddings = self.pretrained_item_representation[pos_items]
        neg_i_g_embeddings = self.pretrained_item_representation[neg_items]

        pos_item_score = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_item_score = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)

        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores


    def get_user_embedding(self, users):
        self.pretrained_user_representation, self.pretrained_item_representation = self.forward(self.norm_adj)
        return self.pretrained_user_representation[users]

    def get_user_aggregated_item_embedding(self, users):
        user_aggregated_item_embeddings = torch.mm(self.interaction_matrix[users], self.pretrained_item_representation)
        return user_aggregated_item_embeddings

    def save_pretrained_representation(self):
        pretrained_user_representation = self.user_embedding.weight.cpu().detach().numpy()
        pretrained_item_representation = self.item_embedding.weight.cpu().detach().numpy()
        with open(os.path.join(self.dataset_path, 'LightGCN_user_representation.npy'), 'wb') as fu:
            np.save(fu, pretrained_user_representation, allow_pickle=True)
        with open(os.path.join(self.dataset_path, 'LightGCN_item_representation.npy'), 'wb') as fi:
            np.save(fi, pretrained_item_representation, allow_pickle=True)


        