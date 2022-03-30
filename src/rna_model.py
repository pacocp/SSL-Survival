import time
import copy
import random
import json
import argparse
import datetime
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn import preprocessing

from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from types_ import *
from utils import *
from losses import *
from read_data import *

class RNAEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List):
        super(RNAEncoder, self).__init__()

        self.in_channels = in_channels

        modules = []
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)

class RNAModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List,
                 out_channels: int):
        super(RNAModel, self).__init__()

        self.encoder = RNAEncoder(in_channels, hidden_dims)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], 200),
            nn.LeakyReLU(),
            nn.Linear(200, out_channels))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

def train_RNA(model, criterion, optimizer, dataloaders,
          save_dir='checkpoints/models/', device=None,
          log_interval=100, summary_writer=None, num_epochs=100, 
          verbose=True):

     since = time.time()

     best_model_wts = copy.deepcopy(model.state_dict())
     best_epoch = 0
     best_loss = 0
     loss_array = {'train': [], 'val': []}
     
     global_summary_step = {'train': 0, 'val': 0}
     scaler = GradScaler()
     for epoch in range(num_epochs):
         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
         print('-' * 10)

         sizes = {'train': 0, 'val': 0}
         inputs_seen = {'train': 0, 'val': 0}
        

         for phase in ['train', 'val']:
            if phase == 'train':
                 model.train()
            else:
                 model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            summary_step = global_summary_step[phase]
            # for logging tensorboard
            last_running_loss = 0.0
            output_list = []
            vital_status_list = []
            survival_months_list = []
            for b_idx, batch in tqdm(enumerate(dataloaders[phase])):
                vital_status_list.append(batch[1].numpy().flatten())
                survival_months_list.append(batch[2].numpy().flatten())
                vital_status = batch[1].to(device)
                survival_months = batch[2].to(device)

                rna_data = batch[0].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(rna_data)
                    output_list.append(outputs.detach().cpu().numpy().flatten())
                    loss = criterion(outputs, vital_status.view(vital_status.size(0)), survival_months.view(survival_months.size(0)))
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                summary_step += 1
                running_loss += loss.item()
                sizes[phase] += rna_data.size(0)
                inputs_seen[phase] += rna_data.size(0)

                # Emptying memory
                del outputs
                torch.cuda.empty_cache()

                if (summary_step % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / inputs_seen[phase]

                    if summary_writer is not None:
                        summary_writer.add_scalar("{}/loss".format(phase), loss_to_log, summary_step)

                    last_running_loss = running_loss
                    last_running_corrects = running_corrects
                    inputs_seen[phase] = 0.0

            global_summary_step[phase] = summary_step
            epoch_loss = running_loss / sizes[phase]

            loss_array[phase].append(epoch_loss)
            output_list = np.concatenate(output_list, axis=0)

            survival_months_list = np.concatenate(survival_months_list, axis=0)
            vital_status_list = np.concatenate(vital_status_list, axis=0)
            CI = get_survival_CI(output_list, survival_months_list, vital_status_list)
            if verbose:
                print('{} Loss: {:.4f}, CI: {:.4f}'.format(
                    phase, epoch_loss, CI))

            if phase == 'val' and CI > best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch

     torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))

     time_elapsed = time.time() - since
     
     model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

     results = {
         'best_epoch': best_epoch,
         'best_loss': best_loss
     }
     return model, results


def evaluate_RNA(model, dataloader, dataset_size, criterion, device=None):
    model.eval()

    running_loss = 0
    running_acc = 0
    output_list = []
    vital_status_list = []
    survival_months_list = []
    for batch in tqdm(dataloader):
        vital_status_list.append(batch[1].numpy().flatten())
        survival_months_list.append(batch[2].numpy().flatten())
        vital_status = batch[1].to(device)
        survival_months = batch[2].to(device)

        rna_data = batch[0].to(device)
        with torch.set_grad_enabled(False):
            outputs = model(rna_data)
            loss = criterion(outputs, vital_status.view(vital_status.size(0)), survival_months.view(survival_months.size(0)))
            running_loss += loss.item()
            output_list.append(outputs.detach().cpu().numpy().flatten())
    
    test_loss = running_loss / dataset_size

    
    output_list = np.concatenate(output_list, axis=0)
    survival_months_list = np.concatenate(survival_months_list, axis=0)
    vital_status_list = np.concatenate(vital_status_list, axis=0)
    
    CI = get_survival_CI(output_list, survival_months_list, vital_status_list)
    print('Loss {}; CI {}'.format(test_loss, CI))
    test_results = {
        'preds': output_list,
        'survival_months': survival_months_list,
        'vital_status': vital_status_list,
        'CI': CI,
        'loss': test_loss
    }

    return test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNA-Seq Training')
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
    parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate to use')
    parser.add_argument('--quick', type=int, default=0,
            help='If a subset of samples is used for a quick training')
    parser.add_argument('--num_epochs', type=int, default=100,
            help='Number of epochs to train the model')
    parser.add_argument('--num_rna_features', type=int, default=17655,
            help='Number of RNA features used as input')
    parser.add_argument('--num_samples', type=int, default=20,
            help='Number of samples to use to train the model')
    parser.add_argument('--problem', type=str, default='survival',
            help='Problem to classify')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as f:
        config = json.load(f)

    print(10*'-')
    print('Config for this experiment \n')
    print(config)
    print(args)
    print(10*'-')

    if 'flag' in config:
        args.flag = config['flag']
    else:
        args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    path_csv = config['path_csv']
    rna_features = args.num_rna_features
    quick = args.quick
    batch_size = args.batch_size
    print('Loading dataset...')

    df = pd.read_csv(path_csv)

    test_CIs = []
    for i in range(10):
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['tcga_project'], random_state=args.seed)

            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['tcga_project'], random_state=args.seed)

            train_df, val_df, test_df, scaler = normalize_dfs(train_df, val_df, test_df)

            train_dataset = RNADataset(train_df, quick=quick, num_samples=args.num_samples, seed=args.seed*i)

            val_dataset = RNADataset(val_df, quick=0, num_samples=64)

            test_dataset = RNADataset(test_df, quick=0, num_samples=64)

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

            if args.problem == 'survival':
                model = RNAModel(in_channels=rna_features,
                                hidden_dims=[6000,512],
                                out_channels=1)
            else:
                model = RNAModel(in_channels=rna_features,
                                hidden_dims=[6000,512],
                                out_channels=len(classes))
            if args.checkpoint is not None:
                model.fc.apply(init_weights_xavier)
            else:
                model.apply(init_weights_xavier)

            if args.checkpoint is not None:
                print('Restoring from checkpoint')
                print(args.checkpoint)
                model.encoder.load_state_dict(torch.load(args.checkpoint))
                print('Loaded model from checkpoint')


            model = model.to(config['device'])
            # add optimizer
            lr = args.lr
            optimizer = AdamW(model.parameters(), weight_decay = 0.1, lr=lr)

            # add loss function
            if args.problem == 'survival':
                criterion = CoxLoss()
            else:
                criterion = nn.CrossEntropyLoss()

            # train model

            if args.log:
                summary_writer = SummaryWriter(
                        os.path.join(config['summary_path'],
                            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_{0}".format(args.flag)))

                summary_writer.add_text('config', str(config))
            else:
                summary_writer = None

            
            model, trainval_results = train_RNA(model, criterion, optimizer, dataloaders,
                                                save_dir=args.save_dir, device='cuda:0',
                                                num_epochs=args.num_epochs)

            test_results = evaluate_RNA(model, test_dataloader, len(test_dataset),
                                        criterion, device='cuda:0')

            test_CIs.append(test_results['CI'])
            with open(args.save_dir+'test_predictions.pkl', 'wb') as f:
                pickle.dump(test_results, f)
        except:
            continue

    print('Mean {}; std {}'.format(np.mean(test_CIs), np.std(test_CIs)))
    print(test_CIs)
