import time
import copy
import random
import datetime
import os

import torch
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

def train_SSL(model, criterion, optimizer, dataloaders,
          save_dir='checkpoints/models/', device=None,
          log_interval=100, summary_writer=None, num_epochs=100, bag=True, verbose=True):

     since = time.time()

     best_model_wts = copy.deepcopy(model.state_dict())
     best_acc = 0.0
     best_epoch = 0
     best_loss = np.inf

     acc_array = {'train': [], 'val': []}
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
            last_running_corrects = 0.0
            last_time = time.time()
            for b_idx, batch in tqdm(enumerate(dataloaders[phase])):
                #batch, labels = permutate_batch(batch, bag)
                labels = batch['labels']
                labels = labels.to(device)
                

                batch['image'].to(device)
                batch['rna_data'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(batch['rna_data'], batch['image'])
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                summary_step += 1
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels)
                sizes[phase] += batch['image'].size(0)
                inputs_seen[phase] += batch['image'].size(0)
                
                # Emptying memory
                del outputs, preds, loss
                torch.cuda.empty_cache()

                if (summary_step % log_interval == 0):
                    loss_to_log = (running_loss - last_running_loss) / inputs_seen[phase]
                    acc_to_log = (running_corrects - last_running_corrects) / inputs_seen[phase]

                    if summary_writer is not None:
                        summary_writer.add_scalar("{}/loss".format(phase), loss_to_log, summary_step)
                        summary_writer.add_scalar("{}/acc".format(phase), acc_to_log, summary_step)

                    last_running_loss = running_loss
                    last_running_corrects = running_corrects
                    inputs_seen[phase] = 0.0

            global_summary_step[phase] = summary_step
            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_corrects.item() / sizes[phase]

            loss_array[phase].append(epoch_loss)
            acc_array[phase].append(epoch_acc)

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                      phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict_best.pt'))
                best_epoch = epoch

     torch.save(model.state_dict(), os.path.join(save_dir, 'model_last.pt'))

     time_elapsed = time.time() - since
     
     model.load_state_dict(torch.load(os.path.join(save_dir, 'model_dict_best.pt')))

     results = {
         'best_acc': best_acc,
         'best_epoch': best_epoch,
         'best_loss': best_loss
     }
     return model, results


def evaluate_SSL(model, dataloader, dataset_size, bag=True, device=None):
    model.eval()

    corrects = 0
    predictions = []
    for batch in tqdm(dataloader):
        labels = batch['label']
        labels = labels.to(device)

        batch['image'].to(device)
        batch['rna_data'].to(device)

        with torch.set_grad_enabled(False):
            outputs = model(batch['rna_data'], batch['image'])
            _, preds = torch.max(outputs, 1)
        
        predictions.append(preds.detach().cpu().numpy())
        corrects += torch.sum(preds == labels)

    accuracy = corrects / dataset_size
    predictions = np.concatenate([predictions], axis=0, dtype=object)

    print('Accuracy of the model {}'.format(accuracy))
    
    return predictions


if __name__ == "__main__":
    batch = []
    for i in range(50):
        dict_inside = {
            'patch_bag': torch.rand((100, 3, 256, 256), dtype=torch.float32),
            'rna_data': torch.rand((56200), dtype=torch.float32)
        }
        batch.append(dict_inside)
    
    batch, labels = permutate_batch(np.asarray(batch), bag=True)
