import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
import json
from fishdataset import FishAirDatasetProcessed
import pandas as pd

def get_dataset_and_dataloader(data_file, img_dir, transform, batch_size, num_workers):
    dataset = FishAirDatasetProcessed(data_file, img_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataset, dataloader

def get_pos_weight(train_file, traits_to_detect = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']):
    assert str(train_file).endswith(".csv"), "File path must end with csv"
    df = pd.read_csv(train_file)
    pos_weight = []
    for trait in traits_to_detect:
        # Need negative count 
        neg_count = sum(df[trait] == 0)
        pos_weight.append(neg_count)

    pos_weight = torch.tensor(pos_weight)
    pos_weight = pos_weight / len(df)

    return pos_weight
        

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'edge')
    
def get_transform(target_size, mean, std, transform_type):
    if transform_type == 'squarepad_augment':
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
        ])
    elif transform_type == 'squarepad_augment_normalize':
        if mean == None or std == None:
            raise Exception('mean or std cannot be None for transforms with normalize')
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomAutocontrast(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'squarepad_no_augment':
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    elif transform_type == 'squarepad_no_augment_normalize':
        if mean == None or std == None:
            raise Exception('mean or std cannot be None for transforms with normalize')
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif transform_type == 'centercrop':
        transform = transforms.Compose([
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]) 
    elif transform_type == 'resize':
        transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),  # Resize the shorter side to target size
            transforms.ToTensor(),
        ])
    else:
        raise Exception('transform_type does not match!')
    return transform
