import torch
import torch.nn as nn
import os
import json
import argparse
import datetime
import numpy as np
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

from resnet import *
from read_data import PatchBagDataset

class AggregationModel(nn.Module):
    def __init__(self, resnet, resnet_dim=2048):
        super(AggregationModel, self).__init__()
        self.resnet = resnet
        self.resnet_dim = resnet_dim

    def forward(self, x):
        (batch_size, bag_size, c, h, w) = x.shape
        x = x.reshape(-1, c, h, w)
        features = self.resnet.forward_extract(x)
        features = features.view(batch_size, bag_size, self.resnet_dim)

        features = features.mean(dim=1)
        return features

def train(model, criterion, optimizer, dataloaders, transforms,
          save_dir='checkpoints/models/', device='cpu',
          log_interval=100, summary_writer=None, num_epochs=100, 
          problem='classification', scheduler=None, verbose=True,
          use_attention=False):
    """ 
    Train classification/regression model.
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            criterion (torch.nn): Loss function
            optimizer (torch.optim): Optimizer
            dataloaders (dict): dict containing training and validation DataLoaders
            transforms (dict): dict containing training and validation transforms
            save_dir (str): directory to save checkpoints and models.
            device (str): device to move models and data to.
            log_interval (int): 
            summary_writer (TensorboardX): to register values into tensorboard
            num_epochs (int): number of epochs of the training
            problem (str): if it is a classification or regresion problem
            verbose (bool): whether or not to display metrics during training
            use_attention (bool): 
        Returns:
            train_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    best_acc = 0.0
    best_epoch = 0
    best_loss = np.inf
    best_outputs = {'train': [], 'val': {}}
    acc_array = {'train': [], 'val': []}
    loss_array = {'train': [], 'val': []}
    
    global_summary_step = {'train': 0, 'val': 0}

    # Creates once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        sizes = {'train': 0, 'val': 0}
        inputs_seen = {'train': 0, 'val': 0}
        running_outputs = {'train': [], 'val': []}
        running_labels = {'train': [], 'val': []}

        for phase in ['train', 'val']:
            if phase == 'train':
                    model.train()
            else:
                    model.eval()

            running_loss = 0.0
            if problem == 'classification' or problem == 'ordinal':
                running_corrects = 0.0
            summary_step = global_summary_step[phase]
            # for logging tensorboard
            last_running_loss = 0.0
            if problem == 'classification' or problem=='ordinal':
                last_running_corrects = 0.0
            for batch in tqdm(dataloaders[phase]):
                wsi = batch[0]
                labels = batch[1]
                size = wsi.size(0)

                if problem == 'classification':
                    labels = labels.flatten()
                labels = labels.to(device)
                wsi = wsi.to(device)
                wsi = transforms[phase](wsi)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    # Casts operations to mixed precision
                    with torch.cuda.amp.autocast():
                        if use_attention:
                            outputs = model(wsi)
                        else:
                            outputs = model(wsi)
                        # saving running outputs
                        running_outputs[phase].append(outputs.detach().cpu().numpy())
                        running_labels[phase].append(labels.cpu().numpy())
                        if problem == 'classification':
                            _, preds = torch.max(outputs,1)
                            
                            loss = criterion(outputs, labels)
                        elif problem == 'regression':
                            loss = criterion(outputs, labels.view(labels.size(0), 1))
                        elif problem == 'ordinal':
                            _, preds = torch.max(outputs,1)
                            loss = criterion(outputs, labels.view(labels.size(0), 1))

                    if phase == 'train':
                        # Scales the loss, and calls backward()
                        # to create scaled gradients
                        scaler.scale(loss).backward()
                        
                        # Unscales gradients and calls
                        # or skips optimizer.step()
                        scaler.step(optimizer)
                        
                        # Updates the scale for next iteration
                        scaler.update()
                        if scheduler is not None:
                            scheduler.step()

                summary_step += 1
                running_loss += loss.item() * wsi.size(0)
                if problem == 'classification' or problem == 'ordinal':
                    running_corrects += torch.sum(preds == labels)
                sizes[phase] += size
                inputs_seen[phase] += size

                # Emptying memory
                outputs = outputs.detach()
                loss = loss.detach()
                torch.cuda.empty_cache()

                if (summary_step % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / inputs_seen[phase]
                    if problem == 'classification' or problem == 'ordinal':
                        acc_to_log = (running_corrects - last_running_corrects) / inputs_seen[phase]

                    if summary_writer is not None:
                        summary_writer.add_scalar("{}/loss".format(phase), loss_to_log, summary_step)
                        if problem == 'classification' or 'problem' == 'ordinal':
                            summary_writer.add_scalar("{}/acc".format(phase), acc_to_log, summary_step)

                    last_running_loss = running_loss
                    if problem == 'classification' or problem == 'ordinal':
                        last_running_corrects = running_corrects
                    inputs_seen[phase] = 0.0

        global_summary_step[phase] = summary_step
        epoch_loss = running_loss / sizes[phase]
        if problem == 'classification' or problem == 'ordinal':
            epoch_acc = running_corrects / sizes[phase]

        loss_array[phase].append(epoch_loss)
        if problem == 'classification' or problem == 'ordinal':
            acc_array[phase].append(epoch_acc)

        if verbose:
            if problem == 'classification' or problem == 'ordinal':
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
            else:
                print('{} Loss: {:.4f}'.format(
                        phase, epoch_loss))
        
        if phase == 'val' and epoch_loss < best_loss:
            if problem == 'classification' or problem == 'ordinal':
                best_acc = epoch_acc
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
            best_epoch = epoch
            best_outputs['val'] = running_outputs['val']
            best_outputs['train'] = running_outputs['train']

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

    results = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'best_outputs_val': best_outputs['val'],
            'best_outputs_train': best_outputs['train'],
            'labels_val': running_labels['val'],
            'labels_train': running_labels['train']
        }

    if problem == 'classification' or problem == 'ordinal':
        results['best_acc'] =  best_acc

    return model, results

def evaluate(model, dataloader, dataset_size, transforms, criterion,
             device='cpu', problem='classification', verbose=True,
             use_attention=False):
    """ 
    Evaluate classification model on test set
        Parameters:
            model (torch.nn.Module): Pytorch model already declared.
            dataloasder (torch.utils.data.DataLoader): dataloader with the dataset
            dataset_size (int): Size of the dataset.
            transforms (torch.nn.Sequential): Transforms to be applied to the data
            device (str): Device to move the data to. Default: cpu.
            problem (str): if it is a classification or regresion problem
            verbose (bool): whether or not to display metrics at the end
            use_attention (bool): whether to use the attention of not
        Returns:
            test_results (dict): dictionary containing the labels, predictions,
                                 probabilities and accuracy of the model on the dataset.
    """
    model.eval()

    corrects = 0
    predictions = []
    probabilities = []
    real = []
    losses = []
    for batch in tqdm(dataloader):        
        wsi = batch[0]
        labels = batch[1]
        
        labels = labels.flatten()
        labels = labels.to(device)

        wsi = wsi.to(device)
        wsi = transforms(wsi)
        with torch.set_grad_enabled(False):
            if use_attention:
                outputs = model(wsi)
            else:
                outputs = model(wsi)

            if problem == 'classification':
                _, preds = torch.max(outputs, 1)
                
                loss = criterion(outputs, labels)
            elif problem == 'regression':
                loss = criterion(outputs, labels.view(labels.size(0), 1))
            elif problem == 'ordinal':
                 _, preds = torch.max(outputs, 1)
                 loss = criterion(outputs, labels.view(labels.size(0), 1))
    
        if problem == 'classification' or problem == 'ordinal':
            predictions.append(preds.detach().to('cpu').numpy())
            corrects += torch.sum(preds == labels)
        probabilities.append(outputs.detach().to('cpu').numpy())
        real.append(labels.detach().to('cpu').numpy())
        losses.append(loss.detach().item())

    if problem == 'classification' or problem == 'ordinal':
        accuracy = corrects / dataset_size
    predictions = np.concatenate([predictions], axis=0, dtype=object)
    probabilities = np.concatenate([probabilities], axis=0, dtype=object)
    real = np.concatenate([real], axis=0, dtype=object)
    if (problem == 'classification' or problem == 'ordinal') and verbose:
        print('Accuracy of the model {}'.format(accuracy))
    else:
        print('Loss of the model {}'.format(np.mean(losses)))
    
    test_results = {
        'outputs': probabilities,
        'real': real
    }

    if problem == 'classification' or problem == 'ordinal':
        test_results['accuracy'] = accuracy.detach().to('cpu').numpy()
        test_results['predictions'] = predictions

    return test_results

if __name__ == "__main__":
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
    
    path_csv = config['path_csv']
    patch_data_path = config['patch_data_path']
    img_size = config['img_size']
    max_patch_per_wsi = config['max_patch_per_wsi']
    rna_features = config['rna_features']
    quick = config.get('quick', None)
    bag_size = config.get('bag_size', 40)
    batch_size = config.get('batch_size', 64)

    if 'flag' in config:
        args.flag = config['flag']
    else:
        args.flag = 'train_{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

    if not os.path.exists(config['save_dir']):
        os.mkdir(config['save_dir'])

    transforms_ = transforms.Compose([
    transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transforms_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print('Loading dataset...')

    df = pd.read_csv(path_csv)

    # here we can decide what to use as label, maybe the status
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Labels'], random_state=args.seed)

    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['Labels'], random_state=args.seed)

    train_dataset = PatchBagDataset(patch_data_path, train_df, img_size,
                         max_patch_per_wsi=max_patch_per_wsi,
                         bag_size=bag_size,
                         transforms=transforms_, quick=quick)
    val_dataset = PatchBagDataset(patch_data_path, val_df, img_size,
                            max_patch_per_wsi=max_patch_per_wsi,
                            bag_size=bag_size,
                            transforms=transforms_val, quick=quick)

    test_dataset = PatchBagDataset(patch_data_path, test_df, img_size,
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

    model = AggregationModel(wsi_encoder)

    if args.checkpoint is not None:
        print('Restoring from checkpoint')
        print(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint))
        print('Loaded model from checkpoint')


    #model = model.cuda(config['device'])
    model = nn.DataParallel(model)
    model.cuda()

    # add optimizer

    lr = config.get('lr', 3e-3)

    optimizer = AdamW(model.parameters(), weight_decay = config['weights_decay'], lr=lr)

    # add loss function
    # loss function in this shit
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