import torch
from torch import nn
from pathlib import Path
import pandas as pd 
import json
import matplotlib.pyplot as plt 
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Torch vision
from torchvision.models import resnet18, ResNet18_Weights

class FishAirDataset(Dataset):
    def __init__(
        self,
        data_file:Path,
        img_dir:Path,
        transform,
        corrupt_file:Path=None,
        use_orig_filename:bool=True
    ):
        self.data_file = data_file
        self.img_dir = img_dir
        self.transform = transform
        if str(self.data_file).endswith('json'):
            data = json.load(data_file.open('r'))
        self.species_id_dict = data['species_id_dict'] # Dict mapping species to ids (labels)
        self.id_species_dict = {v: k for k, v in self.species_id_dict.items()}
        self.num_classes = len(self.species_id_dict)
        #self.samples = data['samples'] # List of samples, where each each sample is a dict 
        if corrupt_file:
            self.samples = self._remove_corrupt_files(data['samples'], corrupt_file, use_orig_filename)
        else:
            self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.use_orig_filename = use_orig_filename
        self.df = pd.DataFrame(self.samples) # Might not be necessary 
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        sample is a dict with keys: ARKID, original_filename, arkid_filename, species_name, label 
        """
        sample = self.samples[idx] 
        if self.use_orig_filename:
            filename = sample['original_filename']
        else:
            filename = sample['arkid_filename']
        
        img_path = self.img_dir / filename
        img = Image.open(img_path)
        img = self.transform(img) # torchvision transform
       
        label = sample['label']
        label_one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        return {
            'image': img,
            'label': label,
            'label_one_hot': label_one_hot
        }

    def _remove_corrupt_files(
        self, 
        samples, 
        corrupt_file:Path, 
        use_orig_filename:bool
    ):
        corrupt_file_data = json.load(corrupt_file.open('r'))
        corrupt_filenames = corrupt_file_data['corrupt_filenames']
        filename_key = 'arkid_filename' if not use_orig_filename else 'original_filename'
        print('Removing corrupt files from dataset')
        final_samples = []
        for sample in tqdm(samples):
            if sample[filename_key] not in corrupt_filenames:
                final_samples.append(sample)
        return final_samples

class FishAirDatasetWithoutImgDir(Dataset):
    def __init__(
        self,
        data_file:Path,
        transform,
        use_orig_filename:bool=False
    ):
        self.data_file = data_file
        self.transform = transform
        if str(self.data_file).endswith('json'):
            data = json.load(data_file.open('r'))
        self.species_id_dict = data['species_id_dict'] # Dict mapping species to ids (labels)
        self.id_species_dict = {v: k for k, v in self.species_id_dict.items()}
        self.num_classes = len(self.species_id_dict)
        #self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.use_orig_filename = use_orig_filename
        self.df = pd.DataFrame(self.samples) # Might not be necessary 
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        sample is a dict with keys: ARKID, original_filename, arkid_filename, species_name, label 
        """
        sample = self.samples[idx] 
        if self.use_orig_filename:
            filename = sample['original_filename']
        else:
            filename = sample['arkid_filename']

        sample_img_dir = Path(sample['img_dir'])
        img_path = sample_img_dir / filename
        img = Image.open(img_path)
        if img.mode != 'RGB':
            print(f"The original image mode for {str(img_path)} is {img.mode}. Converting to RGB...")
            img = img.convert('RGB')
        img = self.transform(img) # torchvision transform
       
        label = sample['label']
        label_one_hot = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        return {
            'image': img,
            'label': label,
            'label_one_hot': label_one_hot
        }

class FishAirDatasetM2m(Dataset):
    def __init__(
        self,
        data_file:Path,
        transform,
        use_orig_filename:bool=False
    ):
        self.data_file = data_file
        self.transform = transform
        if str(self.data_file).endswith('json'):
            data = json.load(data_file.open('r'))
        self.species_id_dict = data['species_id_dict'] # Dict mapping species to ids (labels)
        self.id_species_dict = {v: k for k, v in self.species_id_dict.items()}
        self.num_classes = len(self.species_id_dict)
        #self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.use_orig_filename = use_orig_filename
        self.df = pd.DataFrame(self.samples) # Might not be necessary 
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        sample is a dict with keys: ARKID, original_filename, arkid_filename, species_name, label 
        """
        sample = self.samples[idx] 
        if self.use_orig_filename:
            filename = sample['original_filename']
        else:
            filename = sample['arkid_filename']

        sample_img_dir = Path(sample['img_dir'])
        img_path = sample_img_dir / filename
        img = Image.open(img_path)
        if img.mode != 'RGB':
            print(f"The original image mode for {str(img_path)} is {img.mode}. Converting to RGB...")
            img = img.convert('RGB')
        img = self.transform(img) # torchvision transform
       
        label = sample['label']
        
        return (img, label)

class FishAirDatasetProcessed(Dataset):
    def __init__(
        self,
        data_file:Path,
        img_dir:Path,
        species_column_name:str,
        species_id_dict:dict,
        transform
    ):
        self.data_file = data_file
        self.transform = transform
        self.img_dir = img_dir
        if str(self.data_file).endswith('csv'):
            self.df = pd.read_csv(data_file)
        self.species_id_dict = species_id_dict # Dict mapping species to ids (labels)
        self.id_species_dict = {v: k for k, v in self.species_id_dict.items()}
        self.num_classes = len(self.species_id_dict)
        self.species_column_name = species_column_name
        #self.samples = data['samples'] # List of samples, where each each sample is a dict 
        self.samples = self.df.to_dict('records') # List of samples, where each each sample is a dict 
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        sample is a dict with keys: ARKID, original_filename, arkid_filename, species_name, label 
        """
        sample = self.samples[idx] 
        
        filename = sample['filename']

        sample_img_dir = self.img_dir
        img_path = sample_img_dir / filename
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img) # torchvision transform
       
        label = self.species_id_dict[sample[self.species_column_name]]
        
        return (img, label)

class FishAirDatasetFromArray(Dataset):
    """
    This class takes in a numpy array of images and targets 
    Returns a pytorch dataset 
    No transform is necessary (except possibly changing to a torch tensor)
    """
    def __init__(
        self,
        data,
        targets
    ):
        self.data = data
        self.targets = targets
        assert len(data) == len(targets), "Length of data and targets do not match"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = data[idx]
        label = targets[idx]
    
        return (img, label)
