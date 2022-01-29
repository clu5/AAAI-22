import numpy as np
import torch
from sklearn import metrics


def sort_sum(scores):
    index = scores.argsort(axis=1)[:, ::-1]
    ordered = np.sort(scores, axis=1)[:, ::-1]
    cumsum = np.cumsum(ordered, axis=1) 
    return index, ordered, cumsum


def raps_calibrate(
        scores,
        labels, 
        index, 
        ordered, 
        cumsum, 
        penalty, 
        randomized, 
        allow_zero_sets,
        alpha=0.05,
    ):
    def get_p_value(
            score, 
            label, 
            index, 
            ordered, 
            cumsum, 
            penalty, 
            randomized, 
            allow_zero_sets
        ):
        idx = np.where(index == label)
        tau = cumsum[idx]

        if not randomized:
            return tau + penalty[0]

        e = np.random.random()
        if idx == (0, 0):
            return penalty[0] + (e * tau if allow_zero_sets else tau)
        else:
            p = penalty[0:(idx[1][0] + 1)].sum()
            return p + cumsum[(idx[0], idx[1] - 1)] + (e * ordered[idx])

    mask = -np.ones(scores.shape[0])
    for i in range(mask.shape[0]):
        mask[i] = get_p_value(
        scores[i,:],
        labels[i].item(),
        index[None, i,:],
        ordered[None, i,:],
        cumsum[None, i,:],
        penalty[0, :],
        randomized=True,
        allow_zero_sets=False
    )

    qhat = np.quantile(mask, 1 - alpha, interpolation='higher')
#     qhat = torch.quantile(mask, 1 - alpha)
    return qhat


# Generalized conditional quantile function.
def raps_predict(
        num_class, 
        tau, 
        index, 
        ordered, 
        cumsum, 
        penalty, 
        randomized=True, 
        allow_zero_sets=True,
    ):
    penalty_cumsum = np.cumsum(penalty, axis=1)
    sizes_base = ((cumsum + penalty_cumsum) <= tau).sum(axis=1) + 1
    sizes_base = np.minimum(sizes_base, num_class)

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            j = sizes_base[i] - 1
            cumsum_ij = cumsum[i, j]
            ordered_ij = ordered[i, j]
            p = penalty_cumsum[0, j]
            V[i] = 1 / ordered[i, j] * (tau - (cumsum_ij - ordered_ij) - p)

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau >= 1.0:
        sizes[:] = cumsum.shape[1] 
        # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 
        # allow the user the option to never have empty sets 
        # (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    return [index[i, 0:sizes[i]] for i in range(index.shape[0])]


def get_q_hat(calibration_scores, labels, alpha=0.05):
    if not isinstance(calibration_scores, torch.Tensor):
        calibration_scores = torch.tensor(calibration_scores)
        
    n = calibration_scores.shape[0]
    
    #  sort scores and returns values and index that would sort classes
    values, indices = calibration_scores.sort(dim=1, descending=True)
    
    #  sum up all scores cummulatively and return to original index order 
    cum_scores = values.cumsum(1).gather(1, indices.argsort(1))[range(n), labels]
    
    #  get quantile with small correction for finite sample sizes
    q_hat = torch.quantile(cum_scores, np.ceil((n + 1) * (1 - alpha)) / n)
#     q_hat = np.quantile(cum_scores, np.ceil((n + 1) * (1 - alpha)) / n)
    
    return q_hat


def conformal_inference(scores, q_hat):
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    assert q_hat < 1, 'q_hat must be below 1'
        
    n = scores.shape[0]
    
    values, indices = scores.sort(dim=1, descending=True)
    
    #  number of each confidence prediction set to acheive coverage
    set_sizes = (values.cumsum(1) > q_hat).int().argmax(dim=1)
    
    confidence_sets = [indices[i][0:(set_sizes[i] + 1)] for i in range(n)]
    
    return [x.tolist() for x in confidence_sets]


def validity_metric(true, pred):
    corr = 0
    for t, p in zip(true, pred):
        corr += 1 if t in p else 0
    return corr / len(true)


def top_k_validity_metric(true, pred, k=2):
    corr = 0
    for t, p in zip(true, pred):
        corr += 1 if t in p[:k] else 0
    return corr / len(true)