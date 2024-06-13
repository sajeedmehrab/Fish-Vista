# Setup identification dataset

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

class FishAirDatasetProcessed(Dataset):
    def __init__(
        self,
        data_file:Path,
        img_dir:Path,
        transform,
        traits_to_detect = ['adipose_fin', 'pelvic_fin', 'barbel', 'multiple_dorsal_fin']
    ):
        self.data_file = data_file
        self.transform = transform
        self.img_dir = img_dir
        if str(self.data_file).endswith('csv'):
            self.df = pd.read_csv(data_file)
        self.traits_to_detect = traits_to_detect
        self.num_classes = len(self.traits_to_detect)
        self.samples = self.df.to_dict('records') # List of samples, where each each sample is a dict 
        # adipose_fin,pelvic_fin,barbel,multiple_dorsal_fin
        
        
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
        label = [float(sample[t]) for t in self.traits_to_detect]
        label = np.array(label)
        
        return img, label

    def get_img_filenames(self, indices):
        return [self.samples[i] for i in indices]