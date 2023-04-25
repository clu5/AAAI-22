import collections
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_reduce(func):
    def wrapper(*args, **kws):
        if 'reduce' in kws:
            reduce = kws['reduce']
            if reduce == 'mean': reduce_func = np.mean
            elif reduce == 'median': reduce_func = np.median
            elif reduce == 'max': reduce_func = np.max
            elif reduce == 'min': reduce_func = np.min
            elif reduce == 'var': reduce_func = np.var
            elif reduce == 'std': reduce_func = np.std
            else: raise ValueError(f'{reduce} is not supported')
            kws['reduce'] = reduce_func
        return func(*args,  **kws)
    return wrapper

@get_reduce
def get_naive(df, reduce, num_class=114):
    return df.apply(lambda x: 1 - reduce([x[f'pred_{i}'][0] for i in range(num_class)]), axis=1)

@get_reduce
def get_variance(df, reduce, num_class=114):
    return df.apply(lambda x: reduce([np.var(x[f'pred_{i}']) for i in range(num_class)]), axis=1)

@get_reduce
def get_entropy(df, reduce, num_class=114):
    return df.apply(lambda x: reduce([-(p := np.mean(x[f'pred_{i}'])) * np.log(p) for i in range(num_class)]), axis=1)

def parse(values):
    ret = {}
    image = values['meta']['image']
    label = values['meta']['label']
    subgroup = values['meta']['subgroup']
    ret['image'] = image
    ret['label'] = label
    ret['subgroup'] = subgroup

    #  group mc by class
    class_pred = collections.defaultdict(list)
    for k, v in values.items():
        if k.startswith('mc_'):
            for i, x in enumerate(v[0]):
                class_pred[i].append(x)

    for c, pred in class_pred.items():
        ret[f'pred_{c}'] = pred

    return ret


# Conformal code
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

    return q_hat.item()


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

if __name__ == '__main__':
    run_dir = pathlib.Path('/Users/charles.lu/AAAI/fitz/')
    k = 0
    res = json.load(open(run_dir / f'seed_{k}' / 'test-res.json'))
    df = pd.DataFrame(list(map(parse, res.values())))
    num_class = df.label.unique().shape[0]
    df.insert(3, 'pred', df.apply(
        lambda row: np.array([row[f'pred_{i}']
            for i in range(num_class)]).mean(1).tolist(), axis=1
        )
    )
    df.insert(4, 'naive', get_naive(df, reduce='max'))
    df.insert(5, 'variance', get_variance(df, reduce='mean'))
    df.insert(6, 'entropy', get_entropy(df, reduce='mean'))

    conf_df = df.sample(frac=0.5, replace=False)
    test_df = df[~df.index.isin(conf_df)]

    q_hat = get_q_hat([x for x in conf_df.pred], conf_df.label.values)

    confidence_sets = conformal_inference([x for x in test_df.pred], q_hat)

    test_df.insert(2, 'confidence_set', confidence_sets)
    test_df.insert(3, 'set_size', test_df.confidence_set.map(len))


    fontsize = 24

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(16, 8))
    for i in range(7):
        box = plt.boxplot(
            test_df[test_df.subgroup == i].set_size,
            notch=True,
            positions=[i],
            widths=0.5,
        )
        set_box_color(box, f'C{i}')

    subgroups = list(range(7))
    plt.title('confidence set size by subgroup', fontsize=fontsize)
    plt.xticks(subgroups, fontsize=fontsize)
    plt.xlabel('subgroups', fontsize=fontsize + 4)
    plt.yticks(range(0, 120, 10), fontsize=fontsize)
    plt.ylabel('confidence set size', fontsize=fontsize + 4)
    plt.show()
