import argparse
import itertools
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics


def confusion_mat(pred, label, 
                  label_names=['Fatty', 'Scattered', 'Heterogeneous', 'Dense'],
                  title=None,
                  fontsize=24,
                  save_path=None,
                 ):
    con_mat = metrics.confusion_matrix(label, pred, labels=[1, 2, 3, 4])
    kappa = metrics.cohen_kappa_score(label, pred, weights='linear')
    N = len(label_names)
    plt.clf()
    plt.figure(figsize=(12, 12))
    plt.grid(None)
    title = '' if title is None else title
    print(title)
    plt.title(title + f' Kappa: {kappa:.3f}', fontsize=fontsize)
    plt.xticks(range(N), label_names, fontsize=fontsize - 2)
    plt.yticks(range(N), label_names, fontsize=fontsize - 2)
    plt.xlabel('Prediction', fontsize=fontsize)
    plt.ylabel('Ground truth', fontsize=fontsize)
    plt.imshow(con_mat, cmap=plt.get_cmap('Blues'))
    for i, j in itertools.product(*itertools.tee(range(N))):
        c = 'white' if con_mat[j, i] > con_mat.max() / 2 else 'black'
        plt.text(i, j, f'{con_mat[j, i]}', horizontalalignment='center', fontsize=fontsize, color=c)
        
    if save_path: plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    START = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='path to test results')
    parser.add_argument('-o', '--output', default='.',
                        help='path to output plots')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    outdir = Path(args.output) / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    
#     print(df.head())
    
    white = df.query('race == "white"')
    black = df.query('race == "black or african american"')
    asian = df.query('race == "asian"')
    latin = df.query('race == "hispanic or latino"')
    other = df.query('race == "other or unknown"')
    
    
    confusion_mat(white.density, white.pred, title='white')
    confusion_mat(black.density, black.pred, title='black')
    confusion_mat(asian.density, asian.pred, title='asian')
    confusion_mat(latin.density, latin.pred, title='latin')
    confusion_mat(asian.density, asian.pred, title='other')