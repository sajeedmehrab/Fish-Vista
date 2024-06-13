# Setup train.py
from __future__ import print_function

import csv
import os

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
import json

from config import *

from model import get_custom_model

def train_epoch(model, criterion, optimizer, data_loader):
    model.train()

    train_loss = 0
    total = 0
    all_predicted = []
    all_targets = []
    for inputs, targets in tqdm(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        # outputs, _ = model(normalizer(inputs))
        if 'inception' in MODEL:
            outputs, _ = model(inputs)
        else:
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # breakpoint()

        train_loss += loss.item() * batch_size
        total += batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.sigmoid(outputs) # Get probabilities
        all_predicted.append(predicted.cpu().detach().numpy())
        all_targets.append(targets.cpu().numpy())

    all_predicted = np.vstack(all_predicted)
    all_targets = np.vstack(all_targets)

    average_precisions = []
    for i in range(all_targets.shape[1]):
        ap = average_precision_score(all_targets[:, i], all_predicted[:, i])
        average_precisions.append(ap)
        print(f'{TRAITS_TO_DETECT[i]} - AP: {ap}')

    mean_ap = np.mean(average_precisions)
    
    eps = 0.000001
    msg = f"Training Loss: {train_loss / (total + eps)} | Training AP: {average_precisions} | Training  MAP: {map}"
    print(msg)

    return train_loss / (total + eps), map

def save_checkpoint(map, model, optim, epoch, index=False, checkpoint_stats=None):
    # Save checkpoint.
    print('Saving..')

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'map': map,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if checkpoint_stats:
        for k, v in checkpoint_stats.items():
            state[k] = v

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '_' + str(BASE_FILENAME) + '.t7'
    # ARGS.output_path
    ckpt_path = os.path.join(ARGS.output_path, ckpt_name)
    torch.save(state, ckpt_path)

def save_checkpoint_stats_json(checkpoint_stats, epoch, index=True, type='test'):
    checkpoint_stats['epoch'] = epoch
    if index:
        ckpt_stat_name = 'ckpt_stats_' + type + "_" + BASE_FILENAME + "_" + str(epoch) + '.json'
    else:
        ckpt_stat_name = 'ckpt_stats_' + type + "_" + BASE_FILENAME + "_" + '.json'
    ckpt_stat_path = os.path.join(ARGS.output_path, ckpt_stat_name)
    with open(ckpt_stat_path, 'w') as f:
        json.dump(checkpoint_stats, f)

#######################################################################################


BEST_VAL = 999999  # best validation loss

print(f"==> Building model from custom model: {MODEL}")

# Setup Model
model = get_custom_model(
    model_name=MODEL,
    num_classes=N_CLASSES,
    pretrained=True
)

model = model.to(device)

# Setup optimizer and scheduler
if ARGS.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=ARGS.decay)
elif ARGS.optimizer == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.decay)
else:
    raise NotImplementedError("Given optimizer not implemented")
    
if ARGS.cosine_annealing:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
else:
    scheduler = None

# Check if we need to start from checkpoint
if ARGS.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    if ARGS.checkpoint_path is not None:
        ckpt_t = torch.load(ARGS.checkpoint_path)
        model.load_state_dict(ckpt_t['net'])
        optimizer.load_state_dict(ckpt_t['optimizer'])
        START_EPOCH = ckpt_t['epoch'] + 1

# Set loss function
if ARGS.loss_type == 'BCE':
    criterion = nn.BCEWithLogitsLoss(reduce=True, reduction='mean').to(device)
elif ARGS.loss_type == 'WBCE':
    pos_weight = get_pos_weight(train_file)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduce=True, reduction='mean').to(device)
else:
    raise NotImplementedError("Loss not implemented")
    
# Training loop
for epoch in range(START_EPOCH, EPOCH):
    print(f'Epoch {epoch}')
    # Adjust learning rate:
    #   epoch < 5: linear warmup of learning rate
    #   epoch > 5 and cosine_annealing: cosine annealing scheduler
    #   epoch > 5 and no cosine_annealing: reduce lr linearly beyond epoch 160 and 180
    #       Following LDAM-DRW
    adjust_learning_rate(optimizer, LR, epoch, scheduler)
    
    train_loss, train_map = train_epoch(
        model=model,
        criterion=criterion, 
        optimizer=optimizer,
        data_loader=train_loader
    )
    train_stats = {'train_loss': train_loss, 'train_map': train_map}

    # Validate
    val_eval = evaluate(
        net=model,
        dataloader=val_loader,
        type='Val',
        epoch=epoch
    )
    
    val_loss = val_eval['loss']
    if val_loss <= BEST_VAL:
        BEST_VAL = val_loss
        # test_loader, lv_sp_normal_loader, lv_sp_dif_loader, lv_sp_normal_test_file, lv_sp_difficult_test_file
        checkpoint_stats_k =  ['loss', 'aps', 'map', 'f1_score', 'precision', 'recall', 'roc_aucs', 'mean_roc_auc']

        def _convert_scalar(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x

        test_stats = evaluate(
            net=model,
            dataloader=test_loader,
            type='Test',
            epoch=epoch
        )
        
        test_stats = {k: _convert_scalar(v) for k, v in test_stats.items() if k in checkpoint_stats_k}
        
        all_test_stats = {
            'normal_test_stats': test_stats
        }

        if lv_sp_normal_test_file:
            lv_sp_normal_test_stats = evaluate(
                net=model,
                dataloader=lv_sp_normal_loader,
                type='Test Lv Sp Normal',
                epoch=epoch
            )
            lv_sp_normal_test_stats = {k: _convert_scalar(v) for k, v in lv_sp_normal_test_stats.items() if k in checkpoint_stats_k}
            all_test_stats['leave_sp_normal_test_stats'] = lv_sp_normal_test_stats

        if lv_sp_difficult_test_file:
            lv_sp_dif_test_stats = evaluate(
                net=model,
                dataloader=lv_sp_dif_loader,
                type='Test Lv Sp Difficult',
                epoch=epoch
            )
            lv_sp_dif_test_stats = {k: _convert_scalar(v) for k, v in lv_sp_dif_test_stats.items() if k in checkpoint_stats_k}
            all_test_stats['lv_sp_dif_test_stats'] = lv_sp_dif_test_stats
            

        save_checkpoint(test_stats['map'], model, optimizer, epoch, False, all_test_stats)

        # Save checkpoint stats in json format for ease of viewing
        save_checkpoint_stats_json(all_test_stats, epoch, index=False)

    if WANDB:
        wandb.log({'train_loss': train_stats['train_loss'], 'val_loss': val_eval['loss']}, step=epoch)    
