import collections
import itertools
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def sensitivity_metric(
    true: np.ndarray, 
    prob: np.ndarray, 
    ) -> Dict[str, float]:
    res = {}
    for k in np.unique(true):
        correct = np.sum((prob.argmax(1) == k) & (true == k))
        total = (true == k).sum()
        res[f'{k}_correct'] = correct
        res[f'{k}_total'] = total
        res[f'{k}_sensitivity'] = correct / max(total, 1e-4)
    return res


def kappa_metric(true, prob, weights='linear'):
    kappa = cohen_kappa_score(true, prob, weights=weights)
    return {'kappa': kappa}


def auroc_metric(true, prob, multi_class='ovr'):
    classes = np.unique(true)
    try:
        met = roc_auc_score(true, prob, multi_class=multi_class, labels=sorted(classes))
    except ValueError as e:
        try:  # one vs all
            met = 0
            for k in classes:
                pos_true = np.where(true == k, 1, 0)
                pos_prob = prob[:, k]
                met += roc_auc_score(pos_true, pos_prob)
            met /= classes.shape[0]
        except ValueError as e:
            print('Value error in auroc_metric')
            met = 1e-5
    return met


def predictive_metric(df, 
                      label='label', 
                      pred='confidence_set', 
                      subgroup='subgroup',
                      k=5,
                      labels=list(range(114)),
                     ):
    """ 
    k-set version of predictive parity
    P(Y = t | T = t, A = a) = P(Y = y | T = t, A = b) 
    
    https://arxiv.org/abs/1610.02413
    """
    dd = collections.defaultdict(dict)
    for a in df.loc[:, subgroup].unique():
        sub_df = df[df.subgroup == a]
        for l in labels:
            lab_df = sub_df[sub_df.label == l]
            count = len(lab_df)
            if not count:
                dd[a][l] = None
            else:
                correct = lab_df.loc[:, pred].map(lambda p: l in p[:k]).sum()
                dd[a][l] = correct / count
    return dict(dd)


def pairwise_difference(parity, 
                        subgroups=list(range(6)), 
                        labels=list(range(9))
                       ):
    ret = {}
    for a, b in itertools.combinations(subgroups, 2):
        if a == b:
            continue
        diff = 0
        count = 0
        for l in labels:
            met_a = parity[a][l]
            met_b = parity[b][l]
            if not met_a or not met_b:
                continue
            diff += abs(met_a - met_b)
            count += 1
        ret[a, b] = diff / count

    return ret

