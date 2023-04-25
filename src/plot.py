import argparse
import itertools
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


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


def plot_roc(true, 
             pred, 
             label_map,
             title='', 
             fontsize=32, 
             lw=5,
             figsize=(8, 8),
             save_path=None, 
            ):
    plt.figure(figsize=figsize)
    for i, (k, v) in enumerate(label_map.items()):
        true_i = true[:, i]
        pred_i = pred[:, i]
        fpr, tpr, thresholds = roc_curve(true_i, pred_i)
        plt.plot(fpr, tpr, 
                 lw=lw, alpha=0.7, ls='--', c=f'C{i}',
                 label=f'{k} - {roc_auc_score(true_i, pred_i):.0%}',
                )
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--', alpha=0.5)
    plt.grid(ls='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    auroc = roc_auc_score(true, pred, multi_class='ovr', labels=range(len(label_map)))
    title += f'  {auroc:.1%}'
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.legend(loc="lower right", fontsize=fontsize // 2)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_confusion(true, 
                   pred, 
                   label_map,
                   title='', 
                   fontsize=32, 
                   figsize=(8, 8),
                   save_path=None, 
                  ):
    plt.figure(figsize=figsize)
    con_mat = confusion_matrix(true, pred, labels=range(len(label_map)))
    plt.grid(False)
    plt.imshow(
        con_mat,
        cmap=plt.get_cmap('Blues'),
    )
    labels = list(label_map.keys())
    n = len(labels)
    plt.title(title, fontsize=fontsize)
    plt.xlabel('Predicted', fontsize=fontsize // 2)
    plt.ylabel('Ground truth', fontsize=fontsize // 2)
    plt.xticks(range(len(labels)),labels, fontsize=fontsize // 2)
    plt.yticks(range(len(labels)), labels, fontsize=fontsize // 2)
    for i in range(n):
        for j in range(n):
            color = 'white' if con_mat[j, i] > con_mat.max() / 2 else 'black'
            plt.text(i, j, 
                     f'{con_mat[j, i]:.0f}', 
                     fontsize=fontsize, 
                     color=color, 
                     horizontalalignment='center',
                    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()