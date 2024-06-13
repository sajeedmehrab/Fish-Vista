import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F
import json
from fish_data.dataset import FishAirDataset, FishAirDatasetWithoutImgDir, FishAirDatasetM2m, FishAirDatasetProcessed
import pandas as pd

def get_dataset_and_dataloder(data_file:Path, img_dir:Path, transform, batch_size=16, corrupt_file:Path=None):
    dataset = FishAirDataset(
        data_file,
        img_dir=img_dir,
        transform=transform,
        corrupt_file=corrupt_file
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset, dataloader

def get_dataset_and_dataloder_without_img_dir(data_file:Path, transform, batch_size=16):
    dataset = FishAirDatasetWithoutImgDir(
        data_file,
        transform=transform
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return dataset, dataloader

def get_dataset_and_dataloder_M2m(data_file:Path, transform, batch_size=16):
    dataset = FishAirDatasetM2m(
        data_file,
        transform=transform
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    return dataset, dataloader

def get_dataset_and_dataloader_processed(data_file:Path, img_dir:Path, species_id_dict, transform, species_column_name='standardized_species', batch_size=16, num_workers=8):
    dataset = FishAirDatasetProcessed(
        data_file=data_file,
        img_dir=img_dir,
        species_column_name=species_column_name,
        species_id_dict=species_id_dict,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataset, dataloader


def get_species_id_dict_for_classification(data_file:Path, species_column_name:str):
    """
    Read a csv file and return a species_id dict in ascending order of image count
    """
    assert str(data_file).endswith('.csv')
    df = pd.read_csv(data_file)

    species_img_counts = df[species_column_name].value_counts()
    prev_count = 99999
    species_id_dict = {}
    for i, (species, img_count) in enumerate(species_img_counts.items()):
        assert img_count <= prev_count
        prev_count = img_count
        species_id_dict[species] = i
    
    return species_id_dict


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'edge')
    
def get_transform(target_size, mean, std, transform_type='squarepad'):
    if transform_type == 'squarepad':
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    elif transform_type == 'squarepad_augment':
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
    elif transform_type == 'squarepad_normalize':
        if mean == None or std == None:
            raise Exception('mean or std cannot be None for squarepad_normalize')
        transform = transforms.Compose([
            SquarePad(),
            transforms.Resize(target_size),  # Resize the shorter side to target size
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise Exception('transform_type does not match!')
    return transform

def get_n_samples_per_class_json(data_file:Path):
    data = json.load(data_file.open('r'))
    n_samples_per_class = [0] * len(data['species_id_dict'])
    samples = data['samples']
    for sample in samples:
        label = sample['label']
        n_samples_per_class[label] += 1
    return n_samples_per_class

def get_n_samples_per_class_csv(data_file:Path, species_id_dict:dict, species_column_name:str):
    data_df = pd.read_csv(data_file)
    samples = data_df.to_dict('records')
    n_samples_per_class = [0] * len(species_id_dict)
   
    for sample in samples:
        species = sample[species_column_name]
        label = species_id_dict[species]
        n_samples_per_class[label] += 1
    return n_samples_per_class
