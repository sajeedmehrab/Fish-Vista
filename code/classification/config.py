import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

import json
import os

from data_loader import make_longtailed_imb, get_imbalanced, get_oversampled, get_smote, get_imbalanced_fishair, get_oversampled_fishair, get_smote_fishair, get_processed_fishair

from fish_data.data_setup import get_transform, get_n_samples_per_class_json, get_species_id_dict_for_classification, get_n_samples_per_class_csv
from utils import InputNormalize, sum_t, Logger
import models
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_warmup', default=5, type=int, help='Warm up steps for learning rate')
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet34', 'resnet18', 'resnet50', 'vit_b_32', 'vit_b_16', 'vgg19', 'swin_b', 'inception_v3', 'convnext_base', 'efficientnet_v2_m', 'mobilenet_v3_large', 'maxvit_t', 'resnext50_32x4d', 'cvt_13', 'mobile_vit_xs', 'mobile_vit_v2', 'regnet_y', 'diet_distilled_s', 'pvt_v2', 'swinb_22k'], help='model type (default: ResNet18)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--dataset', required=True,
                        choices=['cifar10', 'cifar100', 'fishair130-bal-50', 'fishair130-imb-low50', 'fishair130-overs-500', 'fishair_processed', 'fishair_processed_bal'], help='Dataset')
    parser.add_argument('--optimizer', required=True,
                        choices=['SGD', 'AdamW'], help='Optimizer to use')
    parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')

    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--sp_id_path', default=None, type=str, help='species_id_dict path if you are resuming')
    parser.add_argument('--net_g', default=None, type=str,
                        help='checkpoint path of network for generation')
    parser.add_argument('--net_g2', default=None, type=str,
                        help='checkpoint path of network for generation')
    parser.add_argument('--net_t', default=None, type=str,
                        help='checkpoint path of network for train')
    parser.add_argument('--net_both', default=None, type=str,
                        help='checkpoint path of both networks')

    parser.add_argument('--beta', default=0.999, type=float, help='Hyper-parameter for rejection/sampling')
    parser.add_argument('--lam', default=0.5, type=float, help='Hyper-parameter for regularization of translation')
    parser.add_argument('--warm', default=160, type=int, help='Deferred strategy for re-balancing')
    parser.add_argument('--gamma', default=0.99, type=float, help='Threshold of the generation')

    parser.add_argument('--eff_beta', default=1.0, type=float, help='Hyper-parameter for effective number')
    parser.add_argument('--focal_gamma', default=1.0, type=float, help='Hyper-parameter for Focal Loss')

    parser.add_argument('--gen', '-gen', action='store_true', help='')
    parser.add_argument('--step_size', default=0.1, type=float, help='')
    parser.add_argument('--attack_iter', default=10, type=int, help='')

    parser.add_argument('--imb_type', default='longtail', type=str,
                        choices=['none', 'longtail', 'step'],
                        help='Type of artificial imbalance')
    parser.add_argument('--loss_type', default='CE', type=str,
                        choices=['CE', 'Focal', 'LDAM'],
                        help='Type of loss for imbalance')
    parser.add_argument('--ratio', default=100, type=int, help='max/min')
    parser.add_argument('--imb_start', default=5, type=int, help='start idx of step imbalance')

    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')
    parser.add_argument('--cost', '-c', action='store_true', help='oversampling')
    parser.add_argument('--effect_over', action='store_true', help='Use effective number in oversampling')
    parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')
    parser.add_argument('--wandb', action='store_true', help='wandb logging')
    parser.add_argument('--cosine_annealing', action='store_true', help='Use cosine annealing')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of dataloader workers')
    parser.add_argument('--server', required=True, type=str, choices=['arc', 'pda', 'arc_fastscratch'])

    return parser.parse_args()


ARGS = parse_args()
if ARGS.seed is not None:
    SEED = ARGS.seed
else:
    SEED = np.random.randint(10000)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATASET = ARGS.dataset
BATCH_SIZE = ARGS.batch_size
MODEL = ARGS.model
WANDB = ARGS.wandb

LR = ARGS.lr
EPOCH = ARGS.epoch
NUM_WORKERS = ARGS.num_workers
START_EPOCH = 0

# LOGFILE_BASE = f"S{SEED}_{ARGS.name}_" \
#     f"L{ARGS.lam}_W{ARGS.warm}_" \
#     f"E{ARGS.step_size}_I{ARGS.attack_iter}_" \
#     f"{DATASET}_R{ARGS.ratio}_{MODEL}_G{ARGS.gamma}_B{ARGS.beta}"

LOGFILE_BASE = f"S{SEED}_{ARGS.name}_" \
    f"{DATASET}_{MODEL}"

LOGNAME = 'Imbalance_' + LOGFILE_BASE
logger = Logger(LOGNAME)
LOGDIR = logger.logdir


# Data
print('==> Preparing data: %s' % DATASET)

if DATASET == 'cifar100':
    N_CLASSES = 100
    N_SAMPLES = 500
    mean = torch.tensor([0.5071, 0.4867, 0.4408])
    std = torch.tensor([0.2675, 0.2565, 0.2761])
elif DATASET == 'cifar10':
    N_CLASSES = 10
    N_SAMPLES = 5000
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
elif DATASET == 'fishair130-bal-50':
    mean = torch.tensor([0.7340, 0.7222, 0.6979])
    std = torch.tensor([0.1397, 0.1602, 0.1874])
    N_CLASSES = 130
    N_SAMPLES = 50
elif DATASET ==  'fishair130-imb-low50':
    mean = torch.tensor([0.7340, 0.7222, 0.6979])
    std = torch.tensor([0.1397, 0.1602, 0.1874])
    N_CLASSES = 130
    # N_SAMPLES = 50
elif DATASET ==  'fishair130-overs-500':
    mean = torch.tensor([0.7340, 0.7222, 0.6979])
    std = torch.tensor([0.1397, 0.1602, 0.1874])
    N_CLASSES = 130
    N_SAMPLES = 500
elif 'fishair_processed' in DATASET:
    mean = torch.tensor([0.8159, 0.7978, 0.7721])
    std = torch.tensor([0.3316, 0.3474, 0.3740])
    N_CLASSES = 419
    # N_SAMPLES = 500
else:
    raise NotImplementedError()

normalizer = InputNormalize(mean, std).to(device)

if 'cifar' in DATASET:
    if ARGS.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
elif 'fishair' in DATASET:
    if 'inception' in MODEL:
        target_size = 299
    elif MODEL == 'efficientnet_v2_m':
        target_size = 480
    else:
        target_size = 224
    if ARGS.augment:
        transform_train = get_transform(
            target_size=target_size, 
            mean=mean, 
            std=std, 
            transform_type='squarepad_augment'
        )
        # No augmentation on test dataset
        transform_test = get_transform(
            target_size=target_size, 
            mean=mean, 
            std=std, 
            transform_type='squarepad'
        )
    else:
        transform_train = get_transform(
            target_size=target_size, 
            mean=mean, 
            std=std, 
            transform_type='squarepad'
        )

        transform_test = transform_train
else:
    raise NotImplementedError()

## Data Loader ##

if 'cifar' in DATASET:
    N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES
    if ARGS.imb_type == 'longtail':
        N_SAMPLES_PER_CLASS_BASE = make_longtailed_imb(N_SAMPLES, N_CLASSES, ARGS.ratio)
    elif ARGS.imb_type == 'step':
        for i in range(ARGS.imb_start, N_CLASSES):
            N_SAMPLES_PER_CLASS_BASE[i] = int(N_SAMPLES * (1 / ARGS.ratio))

    N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)

    train_loader, val_loader, test_loader = get_imbalanced(DATASET, N_SAMPLES_PER_CLASS_BASE, BATCH_SIZE,
                                                        transform_train, transform_test)
# elif 'fishair' in DATASET:
#     if DATASET == 'fishair130-bal-50':
#         N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES

#         train_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/train_bal_50.json')
#         test_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/test_bal_50.json')
#         val_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/val_bal_50.json')

#     elif DATASET == 'fishair130-imb-low50':
#         train_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/train_imbalance_lower_50.json')
#         test_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/test_imbalance_lower_50.json')
#         val_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/val_imbalance_lower_50.json')

#         N_SAMPLES_PER_CLASS_BASE = get_n_samples_per_class_json(train_file)

#     elif DATASET == 'fishair130-overs-500':
#         N_SAMPLES_PER_CLASS_BASE = [int(N_SAMPLES)] * N_CLASSES

#         train_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/train_bal_oversampled_500.json')
#         test_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/test_bal_oversampled_500.json')
#         val_file = Path('/projects/ml4science/FishDatasetTrack/AllData/ClassificationData/val_bal_oversampled_500.json')

#     train_loader, val_loader, test_loader = get_imbalanced_fishair(
#         dataset=DATASET,
#         train_file=train_file, 
#         val_file=val_file, 
#         test_file=test_file, 
#         batch_size=BATCH_SIZE, 
#         TF_train=transform_train, 
#         TF_test=transform_test
#     )

elif 'fishair' in DATASET:
    if DATASET == 'fishair_processed':
        if ARGS.server == 'pda':
            # Set train file path
            train_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_train_final_filtered.csv')
            # Set val file path
            val_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_val_final_filtered.csv')
            # Set test file path
            test_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_test_final_filtered.csv')
            # Set img dir
            img_dir = Path('/data/BGRemovedCropped/all')
            
        elif ARGS.server == 'arc':
             # Set train file path
            train_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Classification/imb_classification_train_final_filtered.csv')
            # Set val file path
            val_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Classification/imb_classification_val_final_filtered.csv')
            # Set test file path
            test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Classification/imb_classification_test_final_filtered.csv')
            # Set img dir
            img_dir = Path('/projects/ml4science/FishAIR/BGRemovedCropped/all')

        elif ARGS.server == 'arc_fastscratch':
            # Set train file path
            train_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_train_final_filtered.csv')
            # Set val file path
            val_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_val_final_filtered.csv')
            # Set test file path
            test_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_test_final_filtered.csv')
            # Set img dir 
            img_dir = Path('/fastscratch/ksmehrab/BGRemovedCropped/all')
        else:
            raise NotImplementedError('Incorrect Server')

        if ARGS.resume:
            if not ARGS.sp_id_path:
                raise Exception("Need to provide species id dict path if resuming")
            with open(ARGS.sp_id_path, 'r') as f:
                species_id_dict = json.load(f)
        else:
            species_id_dict = get_species_id_dict_for_classification(train_file, 'standardized_species')
        # Save the species id dict
        sp_id_filename = 'species_id_dict.json'
        with open(os.path.join(LOGDIR, sp_id_filename), 'w') as f:
            json.dump(species_id_dict, f)
        
        # Set the species column name
        species_column_name = 'standardized_species'
        # transform is set before

        N_SAMPLES_PER_CLASS_BASE = get_n_samples_per_class_csv(
            data_file=train_file, 
            species_id_dict=species_id_dict, 
            species_column_name='standardized_species'
        )

        majority_class_threshold = None
        for label, n_samples in enumerate(N_SAMPLES_PER_CLASS_BASE):
            if n_samples < 500:
                majority_class_threshold = label
                break
                
        minority_class_threshold = None
        for label, n_samples in enumerate(N_SAMPLES_PER_CLASS_BASE):
            if n_samples < 100:
                minority_class_threshold = label
                break

        train_loader, val_loader, test_loader = get_processed_fishair(
            dataset=DATASET, 
            train_file=train_file, 
            val_file=val_file, 
            test_file=test_file, 
            img_dir=img_dir, 
            species_id_dict=species_id_dict, 
            species_column_name=species_column_name, 
            batch_size=BATCH_SIZE, 
            TF_train=transform_train, 
            TF_test=transform_test,
            num_workers=NUM_WORKERS
        )
    elif DATASET == 'fishair_processed_bal':
        if ARGS.server == 'pda':
            #/data/DatasetTrackFinalData/Classification/sampled_bal_classification_train_final_filtered.csv
            # Set train file path
            train_file = Path('/data/DatasetTrackFinalData/Classification/sampled_bal_classification_train_final_filtered.csv')
            # Use train orig file to keep track of majority and minority classes
            train_orig_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_train_final_filtered.csv')
            # Set val file path
            val_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_val_final_filtered.csv')
            # Set test file path
            test_file = Path('/data/DatasetTrackFinalData/Classification/imb_classification_test_final_filtered.csv')
            # Set img dir
            img_dir = Path('/data/BGRemovedCropped/all')
            # Call the method from data_setup.py to get species_id_dict. Use the train file
        elif ARGS.server == 'arc_fastscratch':
            # Set train file path
            train_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/sampled_bal_classification_train_final_filtered.csv')
            # Use train orig file to keep track of majority and minority classes
            train_orig_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_train_final_filtered.csv')
            # Set val file path
            val_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_val_final_filtered.csv')
            # Set test file path
            test_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Classification/imb_classification_test_final_filtered.csv')
            # Set img dir 
            img_dir = Path('/fastscratch/ksmehrab/BGRemovedCropped/all')
        else:
            raise NotImplementedError('Server not implemented for balanced dataset')

        if ARGS.resume:
            if not ARGS.sp_id_path:
                raise Exception("Need to provide species id dict path if resuming")
            with open(ARGS.sp_id_path, 'r') as f:
                species_id_dict = json.load(f)
        else:
            # Use original train file to get species_id_dict
            species_id_dict = get_species_id_dict_for_classification(train_orig_file, 'standardized_species')
        # Save the species id dict
        sp_id_filename = 'species_id_dict.json'
        with open(os.path.join(LOGDIR, sp_id_filename), 'w') as f:
            json.dump(species_id_dict, f)
        
        # Set the species column name
        species_column_name = 'standardized_species'
        # transform is set before

        # Get the original n_samples_per_class. Used for majority and minority species 
        N_SAMPLES_PER_CLASS_BASE = get_n_samples_per_class_csv(
            data_file=train_orig_file, 
            species_id_dict=species_id_dict, 
            species_column_name='standardized_species'
        )

        majority_class_threshold = None
        for label, n_samples in enumerate(N_SAMPLES_PER_CLASS_BASE):
            if n_samples < 500:
                majority_class_threshold = label
                break
                
        minority_class_threshold = None
        for label, n_samples in enumerate(N_SAMPLES_PER_CLASS_BASE):
            if n_samples < 100:
                minority_class_threshold = label
                break

        # After the majority/minority threshold has been set, change the n_samples_per_class
        N_SAMPLES_PER_CLASS_BASE = [50] * len(species_id_dict)

        # Get the train loader based on the balanced train file
        train_loader, val_loader, test_loader = get_processed_fishair(
            dataset=DATASET, 
            train_file=train_file, 
            val_file=val_file, 
            test_file=test_file, 
            img_dir=img_dir, 
            species_id_dict=species_id_dict, 
            species_column_name=species_column_name, 
            batch_size=BATCH_SIZE, 
            TF_train=transform_train, 
            TF_test=transform_test,
            num_workers=NUM_WORKERS
        )
    

N_SAMPLES_PER_CLASS_BASE = tuple(N_SAMPLES_PER_CLASS_BASE)
print("=" * 50)
print()
print(N_SAMPLES_PER_CLASS_BASE)
print()
print("=" * 50)
## To apply effective number for over-sampling or cost-sensitive ##
"""
Wrap this around if/elif for cifar and fishair 
Idea is that, for fishair, the imbalance will be present in the dataset itself
We still want to pass the imbalance flags, to maintain proper flow
"""
if ARGS.over and ARGS.effect_over:
    _beta = ARGS.eff_beta
    effective_num = 1.0 - np.power(_beta, N_SAMPLES_PER_CLASS_BASE)
    N_SAMPLES_PER_CLASS = tuple(np.array(effective_num) / (1 - _beta))
    print(N_SAMPLES_PER_CLASS)
else:
    N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CLASS_BASE

N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS).to(device)

# Method to adjust learning rate
def adjust_learning_rate(optimizer, lr_init, epoch, scheduler):
    lr = lr_init
    if epoch < ARGS.lr_warmup:
        lr = (epoch + 1) * lr_init / ARGS.lr_warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if ARGS.cosine_annealing:
            assert scheduler != None, "Scheduler cannot be None if cosine annealing is set"
            scheduler.step()
        else:
            """decrease the learning rate at 160 and 180 epoch ( from LDAM-DRW, NeurIPS19 )"""
            if epoch >= 160:
                lr /= 100
            if epoch >= 180:
                lr /= 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    


def evaluate(net, dataloader, logger=None):
    is_training = net.training
    net.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct, total = 0.0, 0.0
    major_correct, neutral_correct, minor_correct = 0.0, 0.0, 0.0
    major_total, neutral_total, minor_total = 0.0, 0.0, 0.0

    class_correct = torch.zeros(N_CLASSES)
    class_total = torch.zeros(N_CLASSES)

    all_predicted = []
    all_targets = []

    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        # print(targets)
        # breakpoint()
        # outputs, _ = net(normalizer(inputs))
        outputs = net(normalizer(inputs))
        # outputs, _ = net(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * batch_size
        predicted = outputs[:, :N_CLASSES].max(1)[1]

        all_predicted.extend(predicted.tolist())
        all_targets.extend(targets.tolist())
        
        total += batch_size
        correct_mask = (predicted == targets)
        correct += sum_t(correct_mask)

        # For accuracy of minority / majority classes.
        major_mask = targets < majority_class_threshold
        major_total += sum_t(major_mask)
        major_correct += sum_t(correct_mask * major_mask)

        minor_mask = targets >= minority_class_threshold
        minor_total += sum_t(minor_mask)
        minor_correct += sum_t(correct_mask * minor_mask)

        neutral_mask = ~(major_mask + minor_mask)
        neutral_total += sum_t(neutral_mask)
        neutral_correct += sum_t(correct_mask * neutral_mask)

        for i in range(N_CLASSES):
            class_mask = (targets == i)
            class_total[i] += sum_t(class_mask)
            class_correct[i] += sum_t(correct_mask * class_mask)
    ###############################
    eps = 0.000001
    
    f1 = f1_score(all_targets, all_predicted, average='macro')
    precision = precision_score(all_targets, all_predicted, average='macro')
    recall = recall_score(all_targets, all_predicted, average='macro')

    results = {
        'loss': total_loss / (total + eps),
        'acc': 100. * correct / (total + eps),
        'major_acc': 100. * major_correct / (major_total + eps),
        'neutral_acc': 100. * neutral_correct / (neutral_total + eps),
        'minor_acc': 100. * minor_correct / (minor_total + eps),
        'class_acc': 100. * class_correct / (class_total + eps),
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Major_ACC: %.3f%% | Neutral_ACC: %.3f%% | Minor ACC: %.3f%% | F1: %.3f | Precision: %.3f | Recall: %.3f' % \
          (
              results['loss'], results['acc'], correct, total,
              results['major_acc'], results['neutral_acc'], results['minor_acc'], results['f1_score'], results['precision'], results['recall']
          )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    net.train(is_training)
    return results
