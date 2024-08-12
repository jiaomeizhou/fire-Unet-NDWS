"""
This file is used to train the model.
To run this file, use: python main.py --dir_checkpoint checkpoints
Original code author: Bronte Sihan Li, 2024
Modified by: Jiaomei Zhou, 2024
"""
import torch
import tensorflow as tf
from dataset import *
import logging
import os
from argparse import ArgumentParser
# from model.asufm.asufm import ASUFM
from train import train_next_day_fire, seed_all, load_checkpoint
# from configs.asufm import get_asfum_6_configs
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)


args = parser.parse_args()
seed_all(args.seed)

# Set parameters
limit_features = [
    'elevation',
    'th',
    'vs',
    'tmmn',
    'tmmx',
    'sph',
    'pr',
    'pdsi',
    'NDVI',
    'population',
    'erc',
    'PrevFireMask',
]
# limit_features = [
#     'elevation',
#     'th',
#     'sph',
#     'pr',
#     'NDVI',
#     'PrevFireMask',
# ]
use_bilinear = True
n_channels = len(limit_features)
n_classes = 1
dir_checkpoint = args.dir_checkpoint
os.makedirs(dir_checkpoint, exist_ok=True)
try:
    # Find the last modified checkpoint in the directory
    last_checkpoint = sorted(
        [f for f in os.listdir(dir_checkpoint) if f.startswith('checkpoint')],
        key=lambda f: os.path.getmtime(os.path.join(dir_checkpoint, f)),
        reverse=True,
    )[0]
    print(f'Loading {last_checkpoint}')
    load_model = f'{dir_checkpoint}/{last_checkpoint}'
except IndexError:
    print('No checkpoints found')
    load_model = None
starting_epoch = 15
pos_weight = 3.0
print(f'Starting epoch: {starting_epoch}')
epochs = args.epochs
batch_size = args.batch_size
lr = 0.0001
scale = 0.5
val_percent = 20
amp = True
optimizer = 'adamw'
optimizer_state_dict = None
loss_function = 'bce'
activation = 'relu'
sampling = 'original'
skip_eval = True
use_checkpointing = False

# Datasets
train_data_file_pattern = (
    '../data/northamerica_2012-2023/train/*_ongoing_*.tfrecord'
)
train_data_file_names = tf.io.gfile.glob(train_data_file_pattern)
val_data_file_pattern = (
    '../data/northamerica_2012-2023/val/*_ongoing_*.tfrecord'
)
val_data_file_names = tf.io.gfile.glob(val_data_file_pattern)
print(f'Train data files: {train_data_file_names}')
# Make tf datasets
train_data = tf.data.TFRecordDataset(train_data_file_names)
val_data = tf.data.TFRecordDataset(val_data_file_names)

print(f'Train data: {len(train_data_file_names)} files')
print(f'Val data: {len(val_data_file_names)} files')


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Train the model
# config = get_asfum_6_configs()
# model = ASUFM(config=config, num_classes=1)
vit_name = 'R50-ViT-B_16'
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
model.load_from(weights=np.load(config_vit.pretrained_path))
model = model.to(memory_format=torch.channels_last)

# logging.info(
#     f'Network:\n'
#     f'\t{model.in_chans} input channels\n'
#     f'\t{model.num_classes} output channels (classes)\n'
# )

if load_model:
    state_dict, optimizer_state_dict, _, _, _ = load_checkpoint(load_model)
    model.load_state_dict(state_dict)
    logging.info(f'Model loaded from {load_model}')
    logging.info(f'Optimizer loaded from {load_model}')
    logging.info(f'Starting from Epoch {starting_epoch}')

model.to(device=device)
logging.info(f'Model loaded to {device}')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

train_next_day_fire(
    starting_epoch=starting_epoch,
    train_data=train_data,
    val_data=val_data,
    model=model,
    epochs=epochs,
    dir_checkpoint=dir_checkpoint,
    batch_size=batch_size,
    learning_rate=lr,
    device=device,
    img_scale=scale,
    val_percent=val_percent / 100,
    amp=amp,
    pos_weight=pos_weight,
    limit_features=limit_features,
    optimizer=optimizer,
    optimizer_state_dict=optimizer_state_dict,
    loss_function=loss_function,
    activation=activation,
    sampling_method=sampling,
    skip_eval=skip_eval,
    use_checkpointing=use_checkpointing,
)