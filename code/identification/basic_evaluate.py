import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from model import get_custom_model
from data_setup import get_transform, get_dataset_and_dataloader
from matplotlib import pyplot

from pathlib import Path

import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve

import os
import json
import argparse

## In the script, save the precision-recall curves for each trait as images. Control this with an argument  

def get_optimal_threshold(
    net, dataloader, experiment_name,
    threshold_type = 'pr_curve', 
    show_plot=False,
    plot_save_filepath = None,
    traits_to_detect = ['adipose', 'pelvic', 'barbel', 'dorsal']
):
    is_training = net.training
    net.eval()
    criterion = nn.BCEWithLogitsLoss(reduce=True, reduction='mean')

    total_loss = 0.0
    total = 0.0

    all_predicted = []
    all_targets = []

    for inputs, targets in tqdm(dataloader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.inference_mode():
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
    
    # change column 1 to 1 - all_predicted[1]
    # change column 1 to 1 - all_targets[1]
    all_predicted[:, 1] = 1 - all_predicted[:, 1]
    all_targets[:, 1] = 1 - all_targets[:, 1]
    
    all_testy = []
    all_yhat = []
    for i in range(all_targets.shape[1]):
        if i != 3:
            all_testy.append(all_targets[:, i])
            all_yhat.append(all_predicted[:, i])
        elif i == 3:
            known_mask = all_targets[:, i] != -1.0
            all_testy.append(all_targets[:, i][known_mask])
            all_yhat.append(all_predicted[:, i][known_mask])
            
    if show_plot or plot_save_filepath:  
        fig, axs = pyplot.subplots(1, 4, figsize=(20, 5))
        
    if threshold_type == 'g_means':
        best_thresholds = []
        for i in range(all_targets.shape[1]):
            fpr, tpr, thresholds = roc_curve(all_targets[:, i], all_predicted[:, i])
            gmeans = np.sqrt(tpr * (1-fpr))
            ix = np.argmax(gmeans)
            best_thresholds.append(thresholds[ix])
            
            if show_plot or plot_save_filepath:
                axs[i].plot([0,1], [0,1], linestyle='--', label='No Skill')
                axs[i].plot(fpr, tpr, marker='.', label='Logistic')
                axs[i].scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
                axs[i].set_title(f'{traits_to_detect[i]}')
                # axis labels
                # axs[i].set_xlabel('False Positive Rate')
                # axs[i].set_ylabel('True Positive Rate')
                # axs[i].legend()

    elif threshold_type == 'pr_curve':
        best_thresholds = []
        for i in range(all_targets.shape[1]):
            testy = all_testy[i]
            yhat = all_yhat[i]
            precision, recall, thresholds = precision_recall_curve(testy, yhat)

            # convert to f score
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            best_thresholds.append(thresholds[ix])
           
            if show_plot or plot_save_filepath:
                no_skill = len(testy[testy==1]) / len(testy)
                axs[i].plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
                axs[i].plot(recall, precision, marker='.', label='Logistic')
                axs[i].scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
                axs[i].set_title(f'{traits_to_detect[i]}')
                # axis labels
                # axs[i].set_xlabel('Recall')
                # axs[i].set_ylabel('Precision')
                # axs[i].legend()

    else:
        raise NotImplementedError('Threshold type not implemented')
    
    if show_plot or plot_save_filepath:  
        # Add a single x label and y label for the entire figure
        fig.text(0.5, 0.04, 'Recall', ha='center')
        fig.text(0.04, 0.5, 'Precision', va='center', rotation='vertical')

        # Add a single legend for the entire figure
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', ncol=2)
        pyplot.tight_layout(rect=[0.04, 0.04, 1, 0.96])
        fig.suptitle(f'{experiment_name}')
        if plot_save_filepath:
            assert plot_save_filepath.endswith('.pdf'), "Plt save files must be pdf"
            pyplot.savefig(plot_save_filepath, format='pdf', bbox_inches='tight')
        if show_plot:
            pyplot.show()
        pyplot.close()  
    
    return best_thresholds

def evaluate(
    net, dataloader, type, best_thresholds
):
    is_training = net.training
    net.eval()
    criterion = nn.BCEWithLogitsLoss(reduce=True, reduction='mean')

    total_loss = 0.0
    total = 0.0

    all_predicted = []
    all_targets = []

    for inputs, targets in tqdm(dataloader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.inference_mode():
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
    
    # change column 1 to 1 - all_predicted[1]
    # change column 1 to 1 - all_targets[1]
    all_predicted[:, 1] = 1 - all_predicted[:, 1]
    all_targets[:, 1] = 1 - all_targets[:, 1]
    
    all_testy = []
    all_yhat = []
    for i in range(all_targets.shape[1]):
        if i != 3:
            all_testy.append(all_targets[:, i])
            all_yhat.append(all_predicted[:, i])
        elif i == 3:
            known_mask = all_targets[:, i] != -1.0
            all_testy.append(all_targets[:, i][known_mask])
            all_yhat.append(all_predicted[:, i][known_mask])
    
#     print(all_targets.shape[1])
#     print(sum(known_mask))
#     print(all_testy[0].shape)
    
    average_precisions = []
    for i in range(all_targets.shape[1]):
        ap = average_precision_score(all_testy[i], all_yhat[i], average='macro')
        average_precisions.append(ap)   

#     average_precisions = list(average_precision_score(all_targets, all_predicted, average=None))

    mean_ap = np.mean(average_precisions)
#     print(f'Mean Average Precision: {mean_ap}')
    
    roc_aucs = []
    for i in range(all_targets.shape[1]):
        roc = roc_auc_score(all_testy[i], all_yhat[i], average='macro')
        roc_aucs.append(roc)
    
#     roc_aucs = list(roc_auc_score(all_targets, all_predicted, average=None))

    mean_roc_auc = np.mean(roc_aucs)
    
    # all_preds_threshold = all_predicted >= best_thresholds
    # all_preds_threshold = all_preds_threshold.astype(int)
    f1s_t = []
    precisions_t = []
    recalls_t = []
    for i in range(all_targets.shape[1]):
        preds_threshold = (all_yhat[i] >= best_thresholds[i]).astype(int)
        assert len(np.unique(all_testy[i])) == 2, "Found more than 2 unique elements in in labels"
        f1 = f1_score(all_testy[i], preds_threshold, average='macro')
        f1s_t.append(f1)
        precision = precision_score(all_testy[i], preds_threshold, average='macro')
        precisions_t.append(precision)
        recall = recall_score(all_testy[i], preds_threshold, average='macro')
        recalls_t.append(recall)
    # all_preds_threshold = all_predicted >= 0.5
    # all_preds_threshold = all_preds_threshold.astype(int)
    f1s = []
    precisions = []
    recalls = []
    for i in range(all_targets.shape[1]):
        preds_threshold = (all_yhat[i] >= 0.5).astype(int)
        assert len(np.unique(all_testy[i])) == 2, "Found more than 2 unique elements in in labels"
        f1 = f1_score(all_testy[i], preds_threshold, average='macro')
        f1s.append(f1)
        precision = precision_score(all_testy[i], preds_threshold, average='macro')
        precisions.append(precision)
        recall = recall_score(all_testy[i], preds_threshold, average='macro')
        recalls.append(recall)
          
        
    eps = 0.000001
    results = {
        'loss': total_loss / (total + eps),
        'aps': average_precisions,
        'map': mean_ap,
        'roc_aucs': roc_aucs,
        'mean_roc_auc': mean_roc_auc,
        'f1': f1s,
        'precision': precisions,
        'recall': recalls,
        'f1_t': f1s_t,
        'precision_t': precisions_t,
        'recall_t': recalls_t
    }


    msg = f"{type} | Loss: {total_loss / (total + eps)} | AP: {average_precisions} | MAP: {mean_ap} | ROC_AUCS: {roc_aucs} | mROC_AUC: {mean_roc_auc} | f1: {f1s} | Precs: {precisions} | Recs: {recalls}"
    
    print(msg)

    net.train(is_training)
    return results

def evaluate_all_tests(
    net, all_test_loaders, best_thresholds, 
    save_filepath=None
):
    all_results = {}
    for k, v in all_test_loaders.items():
        print(f'Running on test: {k}')
        results = evaluate(net, v, k, best_thresholds)
        all_results[k] = results
        
    if save_filepath:
        assert save_filepath.endswith('json'), "Save file must be json"
        with open(save_filepath, 'w') as f:
            json.dump(all_results, f)
        
    
    return all_results

parser = argparse.ArgumentParser(description='Trait Identification Evaluation')
parser.add_argument('--model_name', type=str, choices=['resnet34', 'resnet18', 'resnet50', 'vit_b_32', 'vit_b_16', 'vgg19', 'swin_b', 'inception_v3', 'convnext_base', 'efficientnet_v2_m', 'mobilenet_v3_large', 'maxvit_t', 'resnext50_32x4d', 'cvt_13', 'mobile_vit_xs', 'mobile_vit_v2', 'regnet_y', 'diet_distilled_s', 'pvt_v2', 'swinb_22k'], help='model type')
parser.add_argument('--checkpoint_path', default=None, type=str,
                    help='checkpoint path of network for evaluate')
parser.add_argument('--server', default='pda', type=str, choices=['pda', 'arc'], help='Which server we are running on')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--output_path', default=None, type=str,
                    help='path to save all outputs')
parser.add_argument('--name', default='0', type=str, help='name of experiment or run')
parser.add_argument('--num_workers', default=1, type=int, help='Number of dataloader workers')


args = parser.parse_args()

"""
python basic_evaluate.py --model_name resnet18 --checkpoint_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_r18_wbce_basic/ckpt_9229_S9229_tid_r18_basic_fishair_processed_resnet18.t7 --output_path /home/ksmehrab/FishDatasetTrack/Identification/TraitIDBasic/Outputs/results_r18_wbce_basic/ --name resnet18 --server arc --batch_size 256 --num_workers 8
"""

#### CHANGE MODEL AND CHECKPOINT_PATH
MODEL = args.model_name

N_CLASSES = 4

model = get_custom_model(
    model_name=MODEL,
    num_classes=N_CLASSES,
    pretrained=False
)

model = model.to(device)

# Get checkpoint
checkpoint_path = args.checkpoint_path
ckpt_t = torch.load(checkpoint_path)
model.load_state_dict(ckpt_t['net'])
epoch = ckpt_t['epoch']

# Get data loaders
if args.server == 'pda':
#     train_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_train.csv')
    val_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_val.csv')
    test_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_test_inspecies.csv')
    lv_sp_normal_test_file = Path('/data/DatasetTrackFinalData/Identification/trait_identification_test_leavespecies.csv')
    lv_sp_difficult_test_file = None
    img_dir = Path('/data/BGRemovedCropped/all')
elif args.server == 'arc':
#     train_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_train.csv')
    val_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_identification_val.csv')
    test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_id_test_corrected_inspecies.csv')
    lv_sp_normal_test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Identification/trait_id_test_corrected_leavespecies.csv')
    lv_sp_difficult_test_file = Path('/projects/ml4science/FishDatasetTrack/DatasetTrackFinalData/Segmentation/annotations_mlic.csv')
    img_dir = Path('/projects/ml4science/FishAIR/BGRemovedCropped/all')

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

BATCH_SIZE= args.batch_size
num_workers = args.num_workers

test_dataset, test_loader = get_dataset_and_dataloader(
    data_file=test_file,
    img_dir=img_dir,
    transform=test_transform,
    batch_size=BATCH_SIZE,
    num_workers=num_workers
)

val_dataset, val_loader = get_dataset_and_dataloader(
    data_file=val_file,
    img_dir=img_dir,
    transform=test_transform,
    batch_size=BATCH_SIZE,
    num_workers=num_workers
)

if lv_sp_normal_test_file:
    lv_sp_normal_dataset, lv_sp_normal_loader = get_dataset_and_dataloader(
        data_file=lv_sp_normal_test_file,
        img_dir=img_dir,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=num_workers
    )

if lv_sp_difficult_test_file:
    lv_sp_dif_dataset, lv_sp_dif_loader = get_dataset_and_dataloader(
        data_file=lv_sp_difficult_test_file,
        img_dir=img_dir,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        num_workers=num_workers
    )

plot_save_filepath = os.path.join(args.output_path, 'precision_recall_plot.pdf')

best_thresholds = get_optimal_threshold(
    model, val_loader, args.name,
    threshold_type = 'pr_curve', 
    show_plot=False, 
    plot_save_filepath=plot_save_filepath
)

all_test_loaders = {
    'normal_test': test_loader, 
    'leave_species_test': lv_sp_normal_loader,
    'annotated_test': lv_sp_dif_loader
}

save_filename = f'{args.name}_final_eval.json'
save_filepath = os.path.join(args.output_path, save_filename)
all_results = evaluate_all_tests(model, all_test_loaders, best_thresholds, save_filepath)