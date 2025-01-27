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
import scipy.sparse as sp


from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


T = 5

# dspt: discrete soft prompt tuning
# _1: using prompt weight matrix (W) (thus conditioned on user embedding)
# _2: using prompt embedding cancatnated with user embedding as input


# problem: the output layer is totally shared. 

class DSPT1(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(DSPT1, self).__init__(config, dataloader)

        self.base_model = config['recommendation_model']
        self.neg_slope = config['neg_slope']
        self.num_features = len(config['feature_columns'])
        self.embedding_dim = config['embedding_size']

        self.share_dim = config['share_size']
        self.prompt_dim = config['prompt_size']
        

        pretrained_user_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_user_representation.npy'), allow_pickle=True)
        self.user_representation = torch.FloatTensor(pretrained_user_representation).to(self.device)

        pretrained_item_representation = np.load(os.path.join(self.dataset_path, self.base_model + '_item_representation.npy'), allow_pickle=True)
        self.item_representation = torch.FloatTensor(pretrained_item_representation).to(self.device)


        self._init_sensitive_filter(embedding_dim = config['embedding_size'])

        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        if self.base_model == 'BPR':
            self.loss = BPRLoss()
            self.reg_loss = EmbLoss()
            self.forward = self.forward_BPR
            
        elif self.base_model == 'LightGCN':
            self.interaction_matrix_ = dataloader.inter_matrix(form='coo').astype(np.float32)
            self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix_).float().to_dense()
            row_sums = self.interaction_matrix.sum(axis=-1)
            self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
            self.interaction_matrix = self.interaction_matrix.to(self.device)
            self.norm_adj = self.get_adj_mat()
            self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

            self.weight_size = config['weight_size']
            self.n_ui_layers = len(self.weight_size)

            self.loss = BPRLoss()
            self.reg_loss = EmbLoss()

            self.forward = self.forward_LightGCN

        else:
            raise ValueError(f"No implementation for {self.base_model} with DALFD.")



    def calculate_loss(self, interaction, t=T):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        batch_users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        if self.base_model == 'BPR':
            user_embeddings, item_embeddings = self.forward()
            user_e = self.apply_filter(user_embeddings[batch_users, :], t = t)
            pos_e = item_embeddings[pos_items, :]
            neg_e = item_embeddings[neg_items, :]
            pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
            mf_loss = self.loss(pos_item_score, neg_item_score)
            reg_loss = self.reg_loss(user_e, pos_e, neg_e)
            return mf_loss + self.reg_weight * reg_loss, user_e
        elif self.base_model == 'LightGCN':
            self.pretrained_user_representation, self.pretrained_item_representation = self.forward(self.norm_adj)

            u_g_embeddings = self.apply_filter(self.pretrained_user_representation[batch_users], t = t)
            pos_i_g_embeddings = self.pretrained_item_representation[pos_items]
            neg_i_g_embeddings = self.pretrained_item_representation[neg_items]

            pos_item_score = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
            neg_item_score = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)

            mf_loss = self.loss(pos_item_score, neg_item_score)
            reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
            
            return mf_loss + self.reg_weight * reg_loss, u_g_embeddings





    def full_sort_predict(self, interaction, t = T):
        user = interaction[0]
        if self.base_model == 'BPR':
            user_e = self.apply_filter(self.user_representation[user, :], t = t)
            all_item_e = self.item_representation
            score = torch.matmul(user_e, all_item_e.transpose(0, 1))
            return score
        elif self.base_model == 'LightGCN':
            restore_user_e, restore_item_e = self.forward(self.norm_adj)
            u_embeddings = self.apply_filter(restore_user_e[user], t = t)
            scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
            return scores


    def save_pretrained_representation(self):
        return


    def get_user_embedding(self, users, t=T):
        if self.base_model in ['BPR']:
            user_embeddings, _ = self.forward()
        elif self.base_model in ['LightGCN']:
            user_embeddings, _ = self.forward(self.norm_adj)
        else:
            raise ValueError('No implementation.')
        return self.apply_filter(user_embeddings[users], t)



    def _init_sensitive_filter(self, embedding_dim = 64, t=T):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.share_layer = nn.Linear(embedding_dim, self.share_dim, bias=True)
        self.share_layer.apply(init_weights)

        self.prompt_layer_dict = nn.ModuleDict({str(t): nn.Linear(embedding_dim, self.prompt_dim, bias=True) for t in range(1, T+1)})
        for _, layer in self.prompt_layer_dict.items():
            layer.apply(init_weights)

        self.activation = nn.LeakyReLU(self.neg_slope)

        self.out_layer = nn.Linear(self.share_dim + self.prompt_dim, embedding_dim, bias=True)
        self.out_layer.apply(init_weights)

        # temp
        # self.out_layer_dict = nn.ModuleDict({str(t): nn.Linear(self.prompt_dim, embedding_dim, bias=True) for t in range(1, T+1)})
        # for _, layer in self.out_layer_dict.items():
        #     layer.apply(init_weights)

    def apply_filter(self, user_representation, t=T):
        if t == 0:
            return user_representation
        share_embedding = self.share_layer(user_representation)
        prompt_embedding = self.prompt_layer_dict[str(t)](user_representation)
        # activated_embedding = self.activation(prompt_embedding)
        # output = self.out_layer_dict[str(t)](activated_embedding)
        concated_embedding = torch.concat([share_embedding, prompt_embedding], dim=-1)
        activated_concated_embdding = self.activation(concated_embedding)
        output = self.out_layer(activated_concated_embdding)
        return output

    def forward_BPR(self):
        return self.user_representation, self.item_representation

    def forward_LightGCN(self, adj):
        ego_embeddings = torch.cat((self.user_representation, self.item_representation), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


    # LightGCN
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

    # LightGCN
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)