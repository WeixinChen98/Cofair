import torch.nn as nn
import torch
import numpy as np
import os



from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization

class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton


        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.interaction_matrix = self.sparse_mx_to_torch_sparse_tensor(self.interaction_matrix).float().to_dense()
        row_sums = self.interaction_matrix.sum(axis=-1)
        self.interaction_matrix = self.interaction_matrix / row_sums[:, np.newaxis]
        self.interaction_matrix = self.interaction_matrix.to(self.device)

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        return self.user_embedding(user)


    def forward(self):
        return self.user_embedding.weight, self.item_embedding.weight

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def save_pretrained_representation(self):
        pretrained_user_representation = self.user_embedding.weight.cpu().detach().numpy()
        pretrained_item_representation = self.item_embedding.weight.cpu().detach().numpy()
        with open(os.path.join(self.dataset_path, 'BPR_user_representation.npy'), 'wb') as fu:
            np.save(fu, pretrained_user_representation, allow_pickle=True)
        with open(os.path.join(self.dataset_path, 'BPR_item_representation.npy'), 'wb') as fi:
            np.save(fi, pretrained_item_representation, allow_pickle=True)

    def get_fair_representation(self):
        return