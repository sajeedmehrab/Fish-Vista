# Setup config.py
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

import json
import os
import sys
import wandb

from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

from torch.utils.data import Subset, Dataset, DataLoader

from data_setup import get_transform, get_dataset_and_dataloader, get_pos_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
if torch.cuda.is_available():
    N_GPUS = torch.cuda.device_count()
else:
    N_GPUS = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Trait Identification Pipeline')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_warmup', default=5, type=int, help='Warm up steps for learning rate')
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet34', 'resnet18', 'resnet50', 'vit_b_32', 'vit_b_16', 'vgg19', 'swin_b', 'inception_v3', 'convnext_base', 'efficientnet_v2_m', 'mobilenet_v3_large', 'maxvit_t', 'resnext50_32x4d', 'cvt_13', 'mobile_vit_xs', 'mobile_vit_v2', 'regnet_y', 'diet_distilled_s', 'pvt_v2', 'swinb_22k'], help='model type (default: ResNet18)')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--output_path', default=None, type=str,
                        help='path to save all outputs')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--dataset', required=True,
                        choices=['fishair130-bal-50', 'fishair130-imb-low50', 'fishair130-overs-500', 'fishair_processed'], help='Dataset')
    parser.add_argument('--optimizer', required=True,
                        choices=['SGD', 'AdamW'], help='Optimizer to use')
    parser.add_argument('--decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')

    parser.add_argument('--name', default='0', type=str, help='name of experiment or run')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='checkpoint path of network for train')

    parser.add_argument('--focal_gamma', default=1.0, type=float, help='Hyper-parameter for Focal Loss')

    parser.add_argument('--loss_type', default='BCE', type=str,
                        choices=['BCE', 'WBCE', 'Focal'],
                        help='Type of loss for imbalance')

    parser.add_argument('--wandb', action='store_true', help='wandb logging')
    parser.add_argument('--cosine_annealing', action='store_true', help='Use cosine annealing')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of dataloader workers')
    parser.add_argument('--server', default='pda', type=str, choices=['pda', 'arc', 'arc_fastscratch'], help='Which server we are running on')

    # parser.add_argument('--trial', action='store_true', help='trial run for debugging')
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

# Setup logging 
if os.path.exists(ARGS.output_path):
    if 'arc' not in ARGS.server:
        print(f"Output path: {ARGS.output_path} exists. Continuing will overwrite results. Continue? [Y/n]")
        c = input()
        if c == 'Y':
            pass
        elif c == 'n':
            print('Quitting execution')
            sys.exit()
        else:
            print('Invalid selection. Quitting execution')
            sys.exit()
    else:
        print(f"Output path: {ARGS.output_path} exists. Overwriting results")
else:
    os.mkdir(ARGS.output_path)

BASE_FILENAME = f"S{SEED}_{ARGS.name}_" \
    f"{DATASET}_{MODEL}"

if WANDB:
    os.environ['WANDB_DIR'] = str(ARGS.output_path)
    wandb.init(project=BASE_FILENAME)
    config = {
        "model_name": ARGS.name,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCH,
    }
    wandb.config.update(config)
    
    
# Data
print('==> Preparing data: %s' % DATASET)
if DATASET == 'fishair_processed':
    if 'inception' in MODEL:
        target_size = 299
    elif MODEL == 'efficientnet_v2_m':
        target_size = 480
    elif 'mobile_vit' in MODEL:
        target_size = 256
    else:
        target_size = 224
    mean = torch.tensor([0.9353, 0.9175, 0.8923])
    std = torch.tensor([0.1535, 0.1933, 0.2464])
    transform = get_transform(target_size, mean, std, 'squarepad_augment_normalize')
    test_transform = get_transform(target_size, mean, std, 'squarepad_no_augment_normalize')

    if ARGS.server == 'pda':
        train_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_train.csv')
        val_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_val.csv')
        test_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_test_inspecies.csv')
        lv_sp_normal_test_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_test_leavespecies.csv')
        lv_sp_difficult_test_file = None
        img_dir = Path('/data/BGRemovedCropped/all')
    elif ARGS.server == 'arc':
        train_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_train.csv')
        val_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_val.csv')
        test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_test_inspecies.csv')
        lv_sp_normal_test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_test_leavespecies.csv')
        lv_sp_difficult_test_file = None
        img_dir = Path('/projects/ml4science/FishAIR/BGRemovedCropped/all')
    elif ARGS.server == 'arc_fastscratch':
        train_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Identification/trait_identification_train.csv')
        val_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Identification/trait_identification_val.csv')
        test_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Identification/trait_identification_test_inspecies.csv')
        lv_sp_normal_test_file = Path('/fastscratch/ksmehrab/DatasetTrackFinalData/Identification/trait_identification_test_leavespecies.csv')
        lv_sp_difficult_test_file = None
        img_dir = Path('/fastscratch/ksmehrab/BGRemovedCropped/all')
    else:
        raise NotImplementedError('Given server has not been implemented')
    N_CLASSES = 4

else:    
    raise NotImplementedError('Dataset not implemented')

train_dataset, train_loader = get_dataset_and_dataloader(
    data_file=train_file,
    img_dir=img_dir,
    transform=transform,
    batch_size=BATCH_SIZE,
    num_workers=ARGS.num_workers
)

TRAITS_TO_DETECT = train_dataset.traits_to_detect

val_dataset, val_loader = get_dataset_and_dataloader(
    data_file=val_file,
    img_dir=img_dir,
    transform=test_transform,
    batch_size=BATCH_SIZE,
    num_workers=ARGS.num_workers
)

test_dataset, test_loader = get_dataset_and_dataloader(
    data_file=test_file,
    img_dir=img_dir,
    transform=test_transform,
    batch_size=BATCH_SIZE,
    num_workers=ARGS.num_workers
)

if lv_sp_normal_test_file:
    lv_sp_normal_dataset, lv_sp_normal_loader = get_dataset_and_dataloader(
        data_file=lv_sp_normal_test_file,
        img_dir=img_dir,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=ARGS.num_workers
    )

if lv_sp_difficult_test_file:
    lv_sp_dif_dataset, lv_sp_dif_loader = get_dataset_and_dataloader(
        data_file=lv_sp_difficult_test_file,
        img_dir=img_dir,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=ARGS.num_workers
    )

# if ARGS.trial:
#     train_dataset = Subset(train_dataset, list(range(10)))
#     train_loader = DataLoader(train_dataset)
#     val_dataset, val_loader = train_dataset, train_loader
#     test_dataset, test_loader = train_dataset, train_loader
#     lv_sp_normal_dataset, lv_sp_normal_loader = train_dataset, train_loader
#     lv_sp_dif_dataset, lv_sp_dif_loader = train_dataset, train_loader

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

def evaluate(net, dataloader, epoch, type):
    is_training = net.training
    net.eval()
    criterion = nn.BCEWithLogitsLoss(reduce=True, reduction='mean')

    total_loss = 0.0
    total = 0.0

    all_predicted = []
    all_targets = []

    with torch.inference_mode():
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # breakpoint()
            total_loss += loss.item() * batch_size

            predicted = torch.sigmoid(outputs) # Get probabilities
            
            all_predicted.append(predicted.cpu().detach().numpy())
            all_targets.append(targets.cpu().numpy())
            
            total += batch_size

    all_predicted = np.vstack(all_predicted)
    all_targets = np.vstack(all_targets)

    # average_precisions = []
    # for i in range(all_targets.shape[1]):
    #     ap = average_precision_score(all_targets[:, i], all_predicted[:, i], average='macro')
    #     average_precisions.append(ap)
    #     print(f'{TRAITS_TO_DETECT[i]} - AP: {ap}')

    average_precisions = list(average_precision_score(all_targets, all_predicted, average=None))

    mean_ap = np.mean(average_precisions)
    print(f'Mean Average Precision: {mean_ap}')

    roc_aucs = list(roc_auc_score(all_targets, all_predicted, average=None))

    mean_roc_auc = np.mean(roc_aucs)

    all_preds_threshold = all_predicted >= 0.50
    all_preds_threshold = all_preds_threshold.astype(int)

    f1s = []
    precisions = []
    recalls = []
    for i in range(all_targets.shape[1]):
        f1 = f1_score(all_targets[:, i], all_preds_threshold[:, i], average='macro')
        f1s.append(f1)
        precision = precision_score(all_targets[:, i], all_preds_threshold[:, i], average='macro')
        precisions.append(precision)
        recall = recall_score(all_targets[:, i], all_preds_threshold[:, i], average='macro')
        recalls.append(recall)
        
    eps = 0.000001
    results = {
        'loss': total_loss / (total + eps),
        'aps': average_precisions,
        'map': mean_ap,
        'roc_aucs': roc_aucs,
        'mean_roc_auc': mean_roc_auc,
        'f1_score': f1s,
        'precision': precisions,
        'recall': recalls
    }

    msg = f"{type} | Epoch: {epoch}/{EPOCH} | Loss: {total_loss / (total + eps)} | AP: {average_precisions} | MAP: {mean_ap} | ROC_AUC: {mean_roc_auc} | f1: {f1s} | Precs: {precisions} | Recs: {recalls}"
    
    print(msg)

    net.train(is_training)
    return results