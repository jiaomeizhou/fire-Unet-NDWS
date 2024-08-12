"""
This file is used to evaluate the model.
To run this file, use: python evaluate.py --load_model checkpoints/{model_name}.pth
Original code author: Bronte Sihan Li, 2024
Modified by: Jiaomei Zhou, 2024
"""
import os
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NextDayFireDataset
import gc
from argparse import ArgumentParser
from train import (
    load_checkpoint,
    seed_all,
    evaluate,
)

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--load_model', type=str)
args = parser.parse_args()


seed_all(args.seed)

state_dict, _, _, _, input_features = load_checkpoint(args.load_model)

# Set parameters
n_channels = len(input_features)
n_classes = 1
batch_size = 32  # This should not be too small otherwise the class weights might cause index error
amp = False  # This needs to be false for cpu / mps
sampling_method = 'original'
sampling_mode = 'test'
use_augmented = True
save_predictions = True

# Datasets
if use_augmented:
    train_data_file_pattern = (
        '../data/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
    )
    val_data_file_pattern = (
        '../data/northamerica_2012-2023/val/*_ongoing_*.tfrecord'
    )
    test_data_file_pattern = (
        '../data/northamerica_2012-2023/test/*_ongoing_*.tfrecord'
    )
    print('Using augmented dataset')
else:
    train_data_file_pattern = (
        '../data/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
    )
    val_data_file_pattern = (
        '../data/northamerica_2012-2023/val/*_ongoing_*.tfrecord'
    )
    test_data_file_pattern = (
        '../data/northamerica_2012-2023/test/*_ongoing_*.tfrecord'
    )
    print('Using original dataset')
train_data_file_names = tf.io.gfile.glob(train_data_file_pattern)
val_data_file_names = tf.io.gfile.glob(val_data_file_pattern)
test_data_file_names = tf.io.gfile.glob(test_data_file_pattern)

# Make tf datasets
train_data = tf.data.TFRecordDataset(train_data_file_names)
val_data = tf.data.TFRecordDataset(val_data_file_names)
test_data = tf.data.TFRecordDataset(test_data_file_names)

train_set = NextDayFireDataset(
    train_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)

val_set = NextDayFireDataset(
    val_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)

test_set = NextDayFireDataset(
    test_data,
    limit_features_list=input_features,
    clip_normalize=True,
    sampling_method=sampling_method,
    mode=sampling_mode,
)

print("the length of test data: ", len(test_set))

loader_args = dict(
    batch_size=batch_size,
    num_workers=8
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test metrics and loss
train_loader = DataLoader(train_set, shuffle=False, drop_last=False, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)

# Load model
vit_name = 'R50-ViT-B_16' # The pretrained model name
num_classes = 1
n_skip = 3
img_size = 64
vit_patches_size = 16

config_vit = CONFIGS_ViT_seg[vit_name]
config_vit.n_classes = num_classes
config_vit.n_skip = n_skip

if vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

model = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()

model.load_state_dict(state_dict)
model = model.to(memory_format=torch.channels_last)
model = nn.DataParallel(model) # using multiple gpus

dice_score, auc_score, val_loss, precision, recall, f1 = evaluate(
    model,
    val_loader,
    device,
    batch_size,
    n_val=len(test_set),
    save_predictions=save_predictions,
    skip=False,
)

print(f'Result model: {args.load_model}')
print(f'Validation Dice score: {dice_score}')
print(f'Validation AUC score: {auc_score}')
print(f'Validation Loss: {val_loss}')
print(f'Validation Precision: {precision}')
print(f'Validation Recall: {recall}')
print(f'Validation F1 score: {f1}')

gc.collect()