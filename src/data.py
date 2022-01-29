# ==================================================================== #
#                                                                      #
#                         DATASET  / DATALOADER                        #
#                                                                      #
# ==================================================================== #

from pathlib import Path
from typing import (
    Callable, Dict,  # Literal,  # requires python 3.8
    List,
    Optional, Sequence, Union
)
import torch
import torchvision
import monai
import monai.transforms as mtf
import pandas as pd
import numpy as np
from utils import first, index

# SPLIT = Literal['train', 'valid', 'test']
SPLIT = {'train', 'valid', 'test'}

def type_check(inst, inst_type):
    err_msg = f'Got type {type(inst)}. Need {inst_type}.'
    assert isinstance(inst, inst_type), err_msg


class TORCH_DS(torch.utils.data.Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 base: Path,
                 augment: bool = True,
                 transforms = torchvision.transforms.Compose([
                     torchvision.transforms.RandomHorizontalFlip(p=0.5),
                     torchvision.transforms.RandomVerticalFlip(p=0.5),
                     torchvision.transforms.RandomRotation(degrees=30),
                     torchvision.transforms.RandomResizedCrop(size=[224, 224],
                                                              scale=(0.8, 1.2),
                                                              ratio=(0.7, 1.3)),
                     #torchvision.transforms.RandomAffine(degrees=10),
                     torchvision.transforms.RandomPerspective(distortion_scale=0.5, 
                                                              p=0.5),
                 ]),
                 invert_prob: float = 0.3,
                 weight_attr: Optional[str] = None,
                 weight_map: Optional[Dict[str, int]] = None,
                ):
        super().__init__()
        type_check(data, pd.DataFrame)
        type_check(base, Path)
        type_check(augment, bool)
        assert 0 <= invert_prob <= 1
        self.data = data
        self.base = base
        self.augment = augment
        self.transforms = transforms
        self.invert_prob = invert_prob
        self.weight_attr = weight_attr
        self.weight_map = weight_map

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        ret_attr = self.weight_attr
        row = self.data.iloc[index]
        img = row.image
        lab = row.label
        if ret_attr is not None:
            attr = getattr(row, ret_attr)
            if self.weight_map is not None:
                attr = self.weight_map[attr]
        x = torch.tensor(np.moveaxis(np.load(self.base / img), -1, 0))
        if self.augment:
            x = self.random_augment(x)
        x = x.float()
        y = torch.tensor(lab, dtype=torch.long)
        assert not torch.isnan(x).any(), f'NaN issue at index: {index}'
        assert (x == x).all(), f'INF issue in index: {index}'
        return (x, y, attr) if ret_attr else (x, y)

    def random_augment(self, x: torch.Tensor):
        for i in range(1, 3):
            k = torch.randint(low=1, high=20, size=[1])
            x_i = torch.exp(k * x[i])
            x_i -= x_i.mean()
            x_i /= x_i.max() - x_i.min()
            x[i] = x_i

        if torch.rand(size=[1]) < self.invert_prob:
            x *= -1

        x = self.transforms(x)
        return x


class MONAI_DS(monai.data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 example: str = 'image',
                 target: str = 'label',
                 augment: bool = True,
                 transform: Optional[Callable] = None,
                 ):
        data: list = [
            {
                'image': getattr(row, example),
                'label': getattr(row, target),
            } for _, row in df.iterrows()
        ]
        if transform is None:
            transform = [ 
                mtf.LoadImaged('image'),
                mtf.EnsureChannelFirstd('image'),
                mtf.RepeatChanneld('image', 3),
                mtf.ScaleIntensityd('image'),
            ]
            if augment: 
                transform += [
                    mtf.RandFlipd('image', prob=0.5, spatial_axis=[0]),
                    mtf.RandFlipd('image', prob=0.5, spatial_axis=[1]),
                    mtf.RandAffined('image', prob=0.3, 
                                    translate_range=(10, 10),
                                    scale_range=(0.1, 0.1),
                                    shear_range=(0.1, 0.1),
                                    padding_mode='zeros',
                                    mode='bilinear',
                                   ),
                    mtf.RandRotated('image',
                                    prob=0.5,
                                    range_x=1,
                                    mode='bilinear',
                                   ),
                ]
            transform += [
                mtf.ToTensord('image'),
            ]
            transform = mtf.Compose(transform)
        super().__init__(data=data, transform=transform)
        

def get_weighted_sampler(weights: np.ndarray, batch_size: int = 4):
    """
    weights -- list of class weighting (should be same length as dataset)
    """
    
    sampler = torch.utils.data.WeightedRandomSampler(weights, 
                                                     len(weights), 
                                                     replacement=True,
                                                    )
    batch_sampler = torch.utils.data.BatchSampler(sampler, 
                                                  batch_size=batch_size,
                                                  drop_last=True,
                                                 )
    return batch_sampler


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
