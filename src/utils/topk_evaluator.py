# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import ranking_metrics_dict, fairness_metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.ranking_metrics = config['ranking_metrics']
        self.fairness_metrics = config['fairness_metrics']
        self.topk = config['topk']
        self.fairness_topk = config['fairness_topk']
        self.save_recom_result = config['save_recommended_topk']
        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        pos_items_2darray = eval_data.get_eval_items_2darray()
        # the first sensitive attribute would be considered, which is supposed to be binary 
        users_sens = eval_data.get_eval_users_sens().transpose(0, 1)[0]
        assert max(users_sens) == 1
        assert min(users_sens) == 0
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            recommendation_model_name = self.config['recommendation_model']
            fairness_model_name = self.config['fairness_model_model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}_{}-{}-idx{}-top{}-{}.csv'.format(
                recommendation_model_name, fairness_model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        assert len(users_sens) == len(topk_index)
        # if recom right?
        bool_rec_matrix = []
        rec_matrix = []
        for m, n in zip(pos_items, topk_index):
            bool_rec_matrix.append([True if i in m else False for i in n])
        bool_rec_matrix = np.asarray(bool_rec_matrix)

        # get metrics
        metric_dict = {}
        ranking_result_list = self._calculate_ranking_metrics(pos_len_list, bool_rec_matrix)
        for metric, value in zip(self.ranking_metrics, ranking_result_list):
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)


        # fairness_result_list = self._calculate_fairness_metrics(topk_index, users_sens, pos_items, eval_data.dataset.get_user_num(), eval_data.dataset.get_item_num())
        fairness_result_list = self._calculate_fairness_metrics(topk_index, users_sens, pos_items_2darray, eval_data.dataset.get_user_num(), eval_data.dataset.get_item_num())
        for k, v in fairness_result_list.items():
            metric_dict[k] = v
        return metric_dict

    def _check_args(self):
        # Check metrics
        if isinstance(self.ranking_metrics, (str, list)):
            if isinstance(self.ranking_metrics, str):
                self.ranking_metrics = [self.ranking_metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.ranking_metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.ranking_metrics = [metric.lower() for metric in self.ranking_metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_ranking_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.ranking_metrics:
            metric_fuc = ranking_metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)
    
    def _calculate_fairness_metrics(self, topk_items, sens, test_u2i, n_users, n_items):
        result_dict = {}
        for metric in self.fairness_metrics:
            metric_fuc = fairness_metrics_dict[metric.lower()]            
            for k in self.fairness_topk:
                if "_" in metric:
                    metric1, metric2  = metric.split('_')
                    key1 = '{}@{}'.format(metric1, k)
                    key2 = '{}@{}'.format(metric2, k)
                    result_dict[key1], result_dict[key2] = metric_fuc(topk_items, sens, test_u2i, n_users, n_items, k)
                else:
                    key = '{}@{}'.format(metric, k)
                    result_dict[key] = metric_fuc(topk_items, sens, test_u2i, n_users, n_items, k)
        return result_dict

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.ranking_metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
