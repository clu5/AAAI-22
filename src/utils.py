import copy
import functools
import itertools
import json
import os
import pathlib
from typing import (Any, Callable, Dict, 
    Iterable, List, Optional
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision


def first(x: Iterable) -> List:
    """ get first item of an iterable object """
    return next(iter(x.items() if x is isinstance(x, dict) else x))


def index(x: Iterable, k: int = 1) -> List:
    """ get k-th item of iterable object """
    x = x.items() if isinstance(x, dict) else x
    return list(*itertools.islice(x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def get_posterior(row):
    pred = np.array(eval(row.mc_epistemic))
    prob = np.array([softmax(p) for p in pred])
    pred_mean = np.mean(prob, axis=0)
    pred_var = np.mean(np.var(prob, axis=0))
    pred_max = np.max(pred_mean)
    pred = np.argmax(pred_mean)
    entropy = -np.mean(pred_mean * np.log(pred_mean))
    naive = 1 - prob[0].max()
    return {
        'image': row.image,
        'patient': row.patient,
        'race': row.race,
        'age': row.age,
        'label': row.label,
        'density': row.density,
        'area': row.area,
        'laterality': row.laterality,
        'model': row.model,
        'pred': pred,
        'pred_max': pred_max,
        'pred_mean': pred_mean,
        'pred_var': pred_var,
        'entropy': entropy,
        'bc': bc(prob),
        'naive': naive,
    }

def bc(pred):
    m, n = np.argsort(np.mean(pred, axis=0))[-2:]
    h1 = np.histogram(pred[:, m])[0]
    h2 = np.histogram(pred[:, n])[0]
    return sum([np.sqrt(h_i * h_j) for h_i, h_j in zip(h1, h2)]) / 100



def metric_by_race(df, metric, 
                   races=['white', 'black', 'latin', 'asian', 'other'],
                   noprint=False,
                  ):
#     res = {}
    res = {'all': metric(df)}  # all baseline
    for r in races:
        m = metric(df[df.race == r])
        if not noprint:
            print(f'{r}:\t {m:.2f}')
        res[r] = m
    return res

def metric_by_scanner(df, metric, 
                      scanners=['senograph', 'senoscan', 'ads', 'other'],
                      noprint=False,
                     ):
#     res = {}
    res = {'all': metric(df)}  # all baseline
    for r in scanners:
        m = metric(df[df.model== r])
        if not noprint:
            print(f'{r}:\t {m:.2f}')
        res[r] = m
    return res


def exclude_by_variance(df, threshold):
    N = round(len(df) * threshold)
    cutoff = df.pred_var.sort_values(ascending=False).values[N]
    return df[df.pred_var < cutoff]

def exclude_by_entropy(df, threshold):
    N = round(len(df) * threshold)
    cutoff = df.entropy.sort_values(ascending=False).values[N]
    return df[df.entropy < cutoff]

def exclude_by_bc(df, threshold):
    N = round(len(df) * threshold)
    cutoff = df.bc.sort_values(ascending=False).values[N]
    return df[df.bc < cutoff]

def exclude_by_naive(df, threshold):
    N = round(len(df) * threshold)
    cutoff = df.naive.sort_values(ascending=False).values[N]
    return df[df.naive < cutoff]