import pandas as pd
import numpy as np


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
