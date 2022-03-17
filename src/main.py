import os
import json
import argparse
import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split

from model import *
from rna_model import *
from wsi_model import *
from ssl_training import *
from read_data import *
from resnet import resnet50
#from parallel import DataParallelModel, DataParallelCriterion

parser = argparse.ArgumentParser(description='SSL training')
parser.add_argument('--config', type=str, help='JSON config file')
parser.add_argument('--checkpoint', type=str, default=None,
        help='File with the checkpoint to start with')
parser.add_argument('--save_dir', type=str, default=None,
        help='Where to save the checkpoints')
parser.add_argument('--flag', type=str, default=None,
        help='Flag to use for saving the checkpoints')
parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')
parser.add_argument('--log', type=int, default=0,
        help='Use tensorboard for experiment logging')
parser.add_argument('--parallel', type=int, default=0,
        help='Use DataParallel training')
parser.add_argument('--fp16', type=int, default=0,
        help='Use mixed-precision training')
parser.add_argument('--bag_size', type=int, default=50,
                    help='Bag size to use')
parser.add_argument('--max_patch_per_wsi', type=int, default=100,
                    help='Maximum number of paches per wsi')

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

with open(args.config) as f:
    config = json.load(f)

print(10*'-')
print('Config for this experiment \n')
print(config)
print(10*'-')

if 'flag' in config:
    args.flag = config['flag']
else:
    args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

if not os.path.exists(config['save_dir']):
    os.mkdir(config['save_dir'])

path_csv = config['path_csv']
patch_data_path = config['patch_data_path']
img_size = config['img_size']
max_patch_per_wsi = config['max_patch_per_wsi']
rna_features = config['rna_features']
quick = config.get('quick', None)
bag_size = config.get('bag_size', 40)
batch_size = config.get('batch_size', 64)

if 'quick' in config:
    quick = config['quick']
else:
    quick = None

transforms_ = transforms.Compose([
    transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

print('Loading dataset...')

df = pd.read_csv(path_csv)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Labels'], random_state=args.seed)

train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Labels'], random_state=args.seed)

train_df, val_df, test_df = normalize_dfs(train_df, val_df, test_df)

train_dataset = PatchRNADataset(patch_data_path, train_df, img_size,
                         max_patch_per_wsi=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_, quick=quick)
val_dataset = PatchRNADataset(patch_data_path, val_df, img_size,
                         max_patch_per_wsi=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick)

test_dataset = PatchRNADataset(patch_data_path, test_df, img_size,
                         max_patch_per_wsi=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_val, quick=quick)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
            num_workers=config['n_workers'], pin_memory=True, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=config['n_workers'],
pin_memory=True, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
num_workers=config['n_workers'], pin_memory=True, shuffle=False)

dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader}

dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
        }

print('Finished loading dataset and creating dataloader')

print('Initializing models')

wsi_encoder = resnet50(pretrained=True)

layers_to_train = [wsi_encoder.fc, wsi_encoder.layer4, wsi_encoder.layer3]
for param in wsi_encoder.parameters():
    param.requires_grad = False
for layer in layers_to_train:
    for n, param in layer.named_parameters():
        param.requires_grad = True

rna_encoder = RNAEncoder(in_channels=rna_features,
                         hidden_dims=[4096,2048])

model = SSLModel(rna_encoder=rna_encoder,
                    wsi_encoder=wsi_encoder,
                    distance='euclidean',
                    in_channels=1,
                    out_channels=2,
                    hidden_dims=[2])

if args.checkpoint is not None:
    print('Restoring from checkpoint')
    print(args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint))
    print('Loaded model from checkpoint')


#model = model.cuda(config['device'])
model = nn.DataParallel(model)
model.cuda()
# add optimizer
'''
params_to_update = []
wsi_layers = [model.wsi_encoder.resnet.fc, model.wsi_encoder.resnet.layer4, model.wsi_encoder.resnet.layer3]
for layer in wsi_layers:
    for param in layer.named_parameters():
        params_to_update.append(param)

for param in model.rna_encoder.parameters():
    if param.requires_grad:
        params_to_update.append(param)
for layer in [model.fc, model.out_layer]:
    for param in layer.parameters():
        params_to_update.append(param)
'''

lr = config.get('lr', 3e-3)

optimizer = Adam(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

# add loss function
criterion = nn.CrossEntropyLoss()

# train model

if args.log:
    summary_writer = SummaryWriter(
            os.path.join(config['summary_path'],
                datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag)))

    summary_writer.add_text('config', str(config))
else:
    summary_writer = None

model, results = train(model, criterion, optimizer, dataloaders,
              save_dir=config['save_dir'],
              device=config['device'], log_interval=config['log_interval'],
              summary_writer=summary_writer,
              num_epochs=config['num_epochs'])

# test on test set

test_predictions = evaluate(model, test_dataloader, len(test_dataset),device=config['device'])

np.save(config['save_dir']+'test_predictions.npy', test_predictions)
# save results



