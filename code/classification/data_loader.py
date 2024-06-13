import torch
import os
import numpy.random as nr
import numpy as np
import bisect
from PIL import Image
from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from scipy import io

from fish_data.data_setup import get_dataset_and_dataloder_M2m, get_dataset_and_dataloader_processed
from fish_data.dataset import FishAirDatasetFromArray

num_test_samples_cifar10 = [1000] * 10
num_test_samples_cifar100 = [100] * 100

DATA_ROOT = os.path.expanduser('~/data')


def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


def get_oversampled(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building {} CV data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)

    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    train_in_idx = get_oversampled_data(train_cifar, num_sample_per_class)
    train_in_loader = DataLoader(train_cifar, batch_size=batch_size,
                                 sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = DataLoader(test_cifar, batch_size=100,
                            sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = DataLoader(test_cifar, batch_size=100,
                             sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

def get_oversampled_weights_fishair(train_fishair, num_sample_per_class):
    length = train_fishair.__len__()
    num_sample_per_class = list(num_sample_per_class)
    train_in_index = []
    indices = list(range(0,length))
    for i in tqdm(range(0, length)):
        index = indices[i]
        _, label = train_fishair.__getitem__(index)
        train_in_index.append(1 / num_sample_per_class[label])
    
    return train_in_index


def get_oversampled_fishair(dataset, train_file, val_file, test_file, img_dir, num_sample_per_class, species_id_dict, species_column_name, batch_size, TF_train, TF_test, num_workers):
    
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []
    train_fishair, _ = get_dataset_and_dataloader_processed(
        data_file=train_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_train,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )

    ##################################################
    # from torch.utils.data import Subset, DataLoader
    # subset_dataset = Subset(train_fishair, range(0, 3))
    ##################################################
    
    train_in_idx = get_oversampled_weights_fishair(train_fishair, num_sample_per_class)
    # train_in_index = get_oversampled_weights_fishair(subset_dataset, num_sample_per_class)

    train_in_loader = DataLoader(train_fishair, batch_size=batch_size,
                                 sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)),
                                 num_workers=num_workers)

    # train_in_loader = DataLoader(subset_dataset, batch_size=1,
    #                               sampler=WeightedRandomSampler(train_in_index, len(train_in_index)))
                        
    ds.append(train_in_loader)

    test_fishair, test_loader = get_dataset_and_dataloader_processed(
        data_file=test_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_test,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_fishair, val_loader = get_dataset_and_dataloader_processed(
        data_file=val_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_test,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    ds.append(val_loader)
    # ds.append(train_in_loader)
    ds.append(test_loader)
    # ds.append(train_in_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=True, transform=TF_train)
    sample_item = train_cifar.__getitem__(0)
    sample_img, sample_label = sample_item
    print(f'Sample Image Shape: {sample_img.shape}')
    
    train_in_idx = get_imbalanced_data(train_cifar, num_sample_per_class)
    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), num_workers=8)
    ds.append(train_in_loader)

    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx= get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                                  sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

def get_imbalanced_fishair(dataset, train_file, val_file, test_file, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []
    train_fishair, train_in_loader = get_dataset_and_dataloder_M2m(train_file, TF_train, batch_size)
 
    # breakpoint()

    ##################################################
    # from torch.utils.data import Subset, DataLoader
    # subset_dataset = Subset(train_fishair, range(0, 3))
    # subset_dataloader = DataLoader(subset_dataset)
    ##################################################

    ds.append(train_in_loader)
    # ds.append(subset_dataloader)

    test_fishair, test_loader = get_dataset_and_dataloder_M2m(test_file, TF_test, batch_size)
    
    val_fishair, val_loader = get_dataset_and_dataloder_M2m(val_file, TF_test, batch_size)
    
    ds.append(val_loader)
    # ds.append(subset_dataloader)
    ds.append(test_loader)
    # ds.append(subset_dataloader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

def get_processed_fishair(dataset, train_file, val_file, test_file, img_dir, species_id_dict, species_column_name, batch_size, TF_train, TF_test, num_workers):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []
    train_fishair, train_in_loader = get_dataset_and_dataloader_processed(
        data_file=train_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_train,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )

 
    # breakpoint()

    ##################################################
    # from torch.utils.data import Subset, DataLoader
    # subset_dataset = Subset(train_fishair, range(0, 3))
    # subset_dataloader = DataLoader(subset_dataset)
    ##################################################

    ds.append(train_in_loader)
    # ds.append(subset_dataloader)

    test_fishair, test_loader = get_dataset_and_dataloader_processed(
        data_file=test_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_test,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_fishair, val_loader = get_dataset_and_dataloader_processed(
        data_file=val_file,
        img_dir=img_dir, 
        species_id_dict=species_id_dict, 
        transform=TF_test,
        species_column_name=species_column_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    ds.append(val_loader)
    # ds.append(subset_dataloader)
    ds.append(test_loader)
    # ds.append(subset_dataloader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def smote(data, targets, n_class, n_max):
    aug_data = []
    aug_label = []

    for k in range(1, n_class):
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        if class_len > n_max:
            # This class already has more number of images than n_max
            # Continue without augmentation
            print("[MESSAGE] Class {k} has more number of images than n_max")
            continue
        class_dist = np.zeros((class_len, class_len))

        # Augmentation with SMOTE ( k-nearest )
        if smote:
            for i in range(class_len):
                for j in range(class_len):
                    class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
            sorted_idx = np.argsort(class_dist)

            for i in range(n_max - class_len):
                lam = nr.uniform(0, 1)
                row_idx = i % class_len
                col_idx = int((i - row_idx) / class_len) % (class_len - 1)
                new_data = np.round(
                    lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])

                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)

    return np.array(aug_data), np.array(aug_label)


def get_smote(dataset,  num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    if dataset == 'cifar10':
        dataset_ = datasets.CIFAR10
        num_test_samples = num_test_samples_cifar10
    elif dataset == 'cifar100':
        dataset_ = datasets.CIFAR100
        num_test_samples = num_test_samples_cifar100
    else:
        raise NotImplementedError()

    train_cifar = dataset_(root=DATA_ROOT, train=True, download=False, transform=TF_train)

    targets = np.array(train_cifar.targets)
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_cifar.targets = targets[imbal_class_indices]
    train_cifar.data = train_cifar.data[imbal_class_indices]

    assert len(train_cifar.targets) == len(train_cifar.data)

    class_max = max(num_sample_per_class)
    
    # train_cifar.data ==> numpy array, shape (12406, 32, 32, 3)
    # train_cifar.labels ==> numpy array, shape(12406,)
    # nb_classes ==> number of classes, 10 for cifar10
    aug_data, aug_label = smote(train_cifar.data, train_cifar.targets, nb_classes, class_max)

    train_cifar.targets = np.concatenate((train_cifar.targets, aug_label), axis=0)
    train_cifar.data = np.concatenate((train_cifar.data, aug_data), axis=0)

    print("Augmented data num = {}".format(len(aug_label)))
    print(train_cifar.data.shape)

    train_in_loader = torch.utils.data.DataLoader(train_cifar, batch_size=batch_size, shuffle=True, num_workers=16)
    ds.append(train_in_loader)
    print('[BREAKPOINT] SMOTE')
    breakpoint()
    test_cifar = dataset_(root=DATA_ROOT, train=False, download=False, transform=TF_test)
    val_idx, test_idx = get_val_test_data(test_cifar, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_cifar, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds

def get_smote_fishair(dataset, num_sample_per_class, nb_classes, train_file, val_file, test_file, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    train_fishair, train_fishair_loader = get_dataset_and_dataloder_M2m(train_file, TF_train, 128)

    # class_max = max(num_sample_per_class)
    class_max = 10

    n = len(train_fishair)

    # Convert the train data and labels to numpy array for smote sampling
    # print("Getting items for SMOTE sampling")
    # all_samples_loader = DataLoader(train_fishair, batch_size=n)
    # print('Will this work?')
    # all_samples_batch = next(iter(all_samples_loader))
    # train_fishair_data, train_fishair_targets = all_samples_batch

    train_fishair_data = []
    train_fishair_targets = []

    # for imgs, labels in tqdm(train_fishair_loader):
    #     # imgs: (b, 3, 224, 224) -> (b, 224, 224, 3)
    #     train_fishair_data.extend(imgs.numpy().transpose((0, 2, 3, 1)).tolist())
    #     # labels: (b, 1)
    #     train_fishair_targets.extend(labels.reshape(-1).tolist())

    # train_fishair_data = np.array(train_fishair_data)
    # train_fishair_targets = np.array(train_fishair_targets)

    ##################################################
    # from torch.utils.data import Subset, DataLoader
    # subset_dataset = Subset(train_fishair, range(0, 3))
    # subset_dataloader = DataLoader(subset_dataset)
    # n = len(subset_dataset)
    ##################################################

    for i in tqdm(range(n)):
        train_fishair_data.append(subset_dataset.__getitem__(i)[0].numpy().transpose((1, 2, 0)))
        train_fishair_targets.append(subset_dataset.__getitem__(i)[1])
    train_fishair_data = np.stack(train_fishair_data, axis=0)
    train_fishair_targets = np.array(train_fishair_targets)
    
    # Smote sampling
    aug_data, aug_label = smote(train_fishair_data, train_fishair_targets, nb_classes, class_max)

    train_fishair_targets = np.concatenate((train_fishair_targets, aug_label), axis=0)
    train_fishair_data = np.concatenate((train_fishair_data, aug_data), axis=0)

    print("Augmented data num = {}".format(len(aug_label)))
    print(train_fishair_data.shape)

    """
    Create new dataset using the original and augmented data
    """
    train_fishair = FishAirDatasetFromArray(train_fishair_data, train_fishair_targets)
    
    train_in_loader = DataLoader(train_fishair, batch_size=batch_size, shuffle=True, num_workers=8)
    ds.append(train_in_loader)

    print('[BREAKPOINT] SMOTE FISHAIR')
    breakpoint()

    _, test_loader = get_dataset_and_dataloder_M2m(test_file, TF_test, batch_size)
    
    _, val_loader = get_dataset_and_dataloder_M2m(val_file, TF_test, batch_size)
    
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds