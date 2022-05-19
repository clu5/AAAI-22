import pathlib
from typing import Callable, Dict, List, Optional, Sequence, Union
import torch
import torchvision
import monai
import monai.transforms as mtf
import pandas as pd
import numpy as np
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 base: Union[pathlib.Path, str],
                 data: List[Dict],
                 augment: bool = True,
                 transforms = torchvision.transforms.Compose([
                     torchvision.transforms.RandomHorizontalFlip(p=0.5),
                     torchvision.transforms.RandomVerticalFlip(p=0.5),
                     torchvision.transforms.RandomRotation(degrees=30),
                     torchvision.transforms.RandomResizedCrop(size=[224, 224],
                                                              scale=(0.8, 1.2),
                                                              ratio=(0.7, 1.3)),
                     torchvision.transforms.RandomAffine(degrees=10),
                 ])
                ):
        super().__init__()
        self.base = pathlib.Path(base)
        self.data = data
        self.augment = augment
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int, return_meta_data: bool = False):
        example = self.data[index]
        image = example['image']
        label = example['label']
        
        image_path = self.base / image
        if image_path.suffix in ('.png', '.jpg'):
            x = Image.open(image_path)
        elif image_path.suffix == '.npy':
            x = np.load(image_path)
        else:
            raise ValueError(f'Unknown file type {image_path.name}')
            
        x = torchvision.transforms.ToTensor()(x)
        x = torchvision.transforms.CenterCrop((224, 244))(x)
        x = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )(x)
#         x = torch.tensor(np.moveaxis(np.load(self.base / img), -1, 0))
        if self.augment:
            x = self.random_augment(x)
        x = x.float()
        y = torch.tensor(label, dtype=torch.long)
        assert not torch.isnan(x).any(), f'NaN issue at index: {index}'
        assert (x == x).all(), f'INF issue in index: {index}'
        return (x, y, example) if return_meta_data else (x, y)

    def random_augment(self, x: torch.Tensor):
        x = self.transforms(x)
        return x


def get_weighted_sampler(weights: np.ndarray, batch_size: int = 4):
    """
    weights -- list of class weighting (should be same length as dataset)
    """
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, 
        len(weights), 
        replacement=False,
    )
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, 
        batch_size=batch_size,
        drop_last=True,
    )
    return batch_sampler
