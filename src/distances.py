import numpy as np
from scipy.spatial.distance import minkowski, jensenshannon


def histogram_norm(x, bins=50):
    hist, _ = np.histogram(x, bins=bins)
    return hist / sum(hist)


def minkowski_distance(p, q):
    return minkowski(p, q)


def jenson_shannon_distance(p, q):
    return jensenshannon(p, q)


def bhattacharyya_distance(p, q):
    overlap = [np.sqrt(p_i * q_i) for p_i, q_i in zip(p, q)]
    return -np.log(sum(overlap))


    