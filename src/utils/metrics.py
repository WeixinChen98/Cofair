from logging import getLogger
import numpy as np
import scipy


def recall_(pos_index, pos_len):
    # Recall: average single users recall ratio.
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def recall2_(pos_index, pos_len):
    r"""
    All hits are summed up and then averaged for recall.
    :param pos_index:
    :param pos_len:
    :return:
    """
    rec_cum = np.cumsum(pos_index, axis=1)
    rec_ret = rec_cum.sum(axis=0) / pos_len.sum()
    return rec_ret


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.
    :math:`U^{te}` is for all users in the test set.
    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def map_(pos_index, pos_len):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    """
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(np.float), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=np.float)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances
    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relavent to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.
    """
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)

# def dp_jsd_(topk_items, sens, test_u2i, n_users, n_items, topk):
#     rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
#     truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
#     for uid in range(n_users):
#         rank_topk_items[uid][topk_items[uid][:topk]] = 1

#     index1 = (sens == 1)
#     index2 = ~index1

#     rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
#     rank_dis2 = np.sum(rank_topk_items[index2], axis=0)

#     rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
#     return rank_js_distance


# def eopp_jsd_(topk_items, sens, test_u2i, n_users, n_items, topk):
#     rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
#     truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
#     for uid in range(n_users):
#         rank_topk_items[uid][topk_items[uid][:topk]] = 1
#         truth_rank_topk_items[uid][test_u2i[uid]] = 1

#     truth_rank_topk_items = truth_rank_topk_items & rank_topk_items

#     index1 = (sens == 1)
#     index2 = ~index1

#     truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
#     truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

#     truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)
#     return truth_rank_js_distance


def dp_jsd_(topk_items, sens, test_u2i_2darray, n_users, n_items, topk):
    # Create rank_topk_items array more efficiently using advanced indexing
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int8)
    
    # Use advanced indexing to set the top-k items for each user
    user_indices = np.arange(n_users)[:, None]  # Shape: (n_users, 1)
    rank_topk_items[user_indices, topk_items[:, :topk]] = 1

    # Split users based on the sensitive attribute
    index1 = (sens == 1)
    index2 = ~index1

    # Sum over the two groups
    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)

    # Compute Jensen-Shannon distance
    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    return rank_js_distance



def eopp_jsd_(topk_items, sens, test_u2i_2darray, n_users, n_items, topk):
    # Create the rank_topk_items matrix
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)

    # Assign 1s for top-k items using advanced indexing
    user_indices = np.arange(n_users)[:, None]
    rank_topk_items[user_indices, topk_items[:, :topk]] = 1

    # Perform element-wise AND operation
    truth_rank_topk_items = test_u2i_2darray & rank_topk_items

    # Separate the users based on the sensitive attribute (e.g., gender, race)
    index1 = (sens == 1)
    index2 = ~index1

    # Sum over users for each item
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    # Compute Jensen-Shannon distance
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return truth_rank_js_distance


def dp_eopp_(topk_items, sens, test_u2i_2darray, n_users, n_items, topk):
    # Create the rank_topk_items matrix
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)

    # Assign 1s for top-k items using advanced indexing
    user_indices = np.arange(n_users)[:, None]
    rank_topk_items[user_indices, topk_items[:, :topk]] = 1

    # Perform element-wise AND operation
    truth_rank_topk_items = test_u2i_2darray & rank_topk_items

    # Separate the users based on the sensitive attribute (e.g., gender, race)
    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)

    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return rank_js_distance, truth_rank_js_distance

"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
ranking_metrics_dict = {
    'ndcg': ndcg_,
    'recall': recall_,
    'recall2': recall2_,
    'precision': precision_,
    'map': map_,
}


fairness_metrics_dict = {
    'dp': dp_jsd_,
    'eopp': eopp_jsd_,
    'dp_eopp': dp_eopp_,
}

