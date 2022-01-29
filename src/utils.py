import copy
import functools
import itertools
import json
import os
import pathlib
from typing import (Any, Callable, Dict, 
    Iterable, List, Optional
)

from sklearn.metrics import cohen_kappa_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import monai
import torchvision


def first(x: Iterable) -> List:
    """ get first item of an iterable object """
    return next(iter(x.items() if x is isinstance(x, dict) else x))


def index(x: Iterable, k: int = 1) -> List:
    """ get k-th item of iterable object """
    x = x.items() if isinstance(x, dict) else x
    return list(*itertools.islice(x))


def inv_weight_freq(df: pd.DataFrame, column: str = 'label'):
    """ weight loss by inverse class frequency """
    N = df.shape[0]
    prop = {k: N / v for k, v in getattr(df, column).value_counts().to_dict().items()}
    M = sum(prop.values())
    return {k: v / M for k, v in prop.items()}


def sensitivity_metric(pred: np.ndarray, 
                       label: np.ndarray, 
                       label_map: Dict):
    res = {}
    for k, v in label_map.items():
        corr = pred == label
        res[f'{k}_correct'] = (v == label[corr]).sum()
        res[f'{k}_total'] = (v == label).sum()
        res[f'{k}_sensitivity'] = res[f'{k}_correct'] / max(res[f'{k}_total'], 1e-4)
    return res


def kappa_metric(true, pred, weights='linear'):
    kappa = cohen_kappa_score(true, pred, weights=weights)
    return {'kappa': kappa}


def auroc_metric(true, pred, multi_class='ovr', labels=[0, 1, 2]):
    try:
        met = roc_auc_score(true, pred, multi_class=multi_class, labels=labels)
    except ValueError as e:
        try:
            # try only combining plus and pre plus classes
            pos_true = np.where(true >= 1, 1, 0)
            pos_pred = pred[:, 1:].sum(1)
            met = roc_auc_score(pos_true, pos_pred)
        except ValueError as e:
            met = 0.0001
    return met


def get_model(arch: str = 'resnet18', 
              pretrained: bool = False, 
              in_channels: int = 3, 
              out_channels: int = 1,
              spatial_dims: int = 2,
              dropout_prob: float = 0.1, 
             ):
#     if arch in monai.networks.nets.__dir__():
#         model = getattr(monai.networks.nets, arch)(
#             pretrained=pretrained,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             spatial_dims=spatial_dims,
#             dropout_prob=dropout_prob,
#         )
    if arch in torchvision.models.__dir__():
        model = getattr(torchvision.models, arch)(pretrained=pretrained)
        if 'fc' in dir(model):
            model.fc = torch.nn.Linear(model.fc.in_features, out_channels)
        elif 'classifier' in dir(model):
            model.classifier = torch.nn.Linear(
                model.classifier.in_features,
                out_channels
            )
        else:
            SystemExit(f'Cannot modify output layer')
    else:
        SystemExit(f'{arch} is not a recognized model architecture')

    return model


def monte_carlo_it(model, input_data, n_it=10):
    """Performs monte carlo iterations for n_it"""
    model = copy.deepcopy(model)
    pred_lst = []

    with torch.no_grad():
        # Activate dropout
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        for mc_it in range(n_it):
            pred = model(input_data)
            pred_lst.append(pred.detach().cpu().numpy().tolist())

    return pred_lst


def epoch_loop(trainer, param, run_dir, logger=None, epochs=None):
    best_score = float('-inf')
    best_epoch = 0
    patience = 0
    train_loss, valid_loss = [], []
    train_metric, valid_metric = [], []
    epochs = param['epochs'] if epochs is None else epochs
    for e in range(epochs):
        train_res = trainer.train()
        trn_loss, trn_met = train_res['loss'], train_res['metric']
        train_loss.append(trn_loss)
        train_metric.append(trn_met)
        valid_res = trainer.validate()
        val_loss, val_met = valid_res['loss'], valid_res['metric']
        valid_loss.append(val_loss)
        valid_metric.append(val_met)
        score = round(val_met, 4)
        if score > best_score:
            best_score = score
            best_epoch = e
            patience = 0
            logger.info(f'    Saved checkpoint ==> {best_score}')
            model_name = f'checkpoint-{best_score}-{str(e).zfill(3)}.pth'
            trainer.save(run_dir / model_name, epoch=e)
        else:
            patience += 1
        if patience > param['early_stop']:
            logger.info(f'    Stopped at {e}')
            break
    else:
        logger.info('    Finished training')
    logger.info(f'    Best checkpoint epoch {best_epoch} - {best_score}')
        
    return {
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'train_metric': train_metric,
        'valid_metric': valid_metric,
    }
    
    
def plot_curve(train, valid, 
               train_label=None, valid_label=None,
               fontsize=16,
               ylim=None,
               title=None,
               save_path=None,
              ):
    if title: plt.title(title)
    if ylim: plt.ylim(*ylim)
    plt.plot(train, label=train_label)
    plt.plot(valid, label=valid_label)
    plt.legend(fontsize=fontsize)
    if save_path:
        plt.savefig(save_path)
    plt.clf()