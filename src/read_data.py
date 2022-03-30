import os
import random
import pickle

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lmdb
import lz4framed
import cv2

from types_ import *

random.seed(10)

class InvalidFileException(Exception):
    pass

def decompress_and_deserialize(lmdb_value: Any):
    try:
        img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
    except:
        return None
    image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
    image = np.copy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2,0,1)

class PatchBagDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None, bag_size=40,
            max_patches_total=300, quick=False, num_samples=20):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.bag_size = bag_size
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.index = []
        self.data = {}
        self.num_samples = num_samples
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(self.num_samples)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()

            # get WSI and labels
            WSI = row['wsi_file_name']
            survival_months = row['survival_months']
            status = row['status']

            project = row['tcga_project'] 

            if not os.path.exists(os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI)): 
                print('Not exist {}'.format(os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI)))
                continue
            
            # get patches and keys from lmdb

            path = os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                print(e)
                continue

            # fill self.data 
            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            images = random.sample(n_patches, n_selected)
            self.data[WSI] = {w.lower(): row[w] for w in row.keys()}
            self.data[WSI].update({'WSI': WSI, 'images': images, 'n_images': len(images), 
                                   'lmdb_path': path, 'keys': keys})

            # fill self.index
            for k in range(len(images) // self.bag_size):
                self.index.append((WSI, self.bag_size * k, survival_months, status))
            #import pdb; pdb.set_trace()

    def shuffle(self):
        for k in self.data.keys():
            wsi_row = self.data[k]
            np.random.shuffle(wsi_row['images'])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except:
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        return torch.from_numpy(image).permute(2,0,1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        (WSI, i, survival_months, status) = self.index[idx]
        imgs = []
        row = self.data[WSI]
        lmdb_connection = lmdb.open(row['lmdb_path'],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        with lmdb_connection.begin(write=False) as txn:
            for patch in row['images'][i:i + self.bag_size]:
                lmdb_value = txn.get(row['keys'][patch])
                img = self.decompress_and_deserialize(lmdb_value)
                imgs.append(img)

        img = torch.stack(imgs, dim=0)
        return img, survival_months, status

class PatchDataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None,
            max_patches_total=300, quick=False, le=None):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.keys = []
        self.images = []
        self.filenames = []
        self.labels = []
        self.lmdbs_path = []
        self.le = le
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
            csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
            csv_file['labels'] = [0] * csv_file.shape[0]
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            row = row.to_dict()
            WSI = row['wsi_file_name']
            data_path = row['patch_data_path']
            label = np.asarray(row['labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
            label = torch.tensor(label, dtype=torch.float32)
            #label = label.flatten()
            try:
                path = os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))
                
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
           
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
                #n_patches = sum(1 for _ in open(os.path.join(data_path, WSI, 'loc.txt'))) - 2
                n_selected = min(n_patches, self.max_patches_total)
                n_patches= list(range(n_patches))
                n_patches_index = random.sample(n_patches, n_selected)
                '''
                n_patches_index = []
                for idx in n_patches_index_aux:
                    lmdb_value = lmdb_txn.get(keys[idx])
                    try:
                        img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
                    except:
                        continue

                    n_patches_index.append(idx)
                '''
            except:
                print('Error with db {}'.format(os.path.join(data_path, WSI, WSI.replace('.svs', '.db'))))
                continue
            #self.keys.append(keys)
            #self.random_index.append(n_patches_index)

            for i in n_patches_index:
                #self.images.append(os.path.join(data_path, WSI, WSI + '_patch_{}.png'.format(i)))
                self.images.append(i)
                self.filenames.append(WSI)
                self.labels.append(label)
                self.lmdbs_path.append(path)
                self.keys.append(keys[i])

    def decompress_and_deserialize(self, lmdb_value: Any):
        try:
            img_name, img_arr, img_shape = pickle.loads(lz4framed.decompress(lmdb_value))
        except Exception as e:
            print(e)
            return None
        image = np.frombuffer(img_arr, dtype=np.uint8).reshape(img_shape)
        image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2,0,1)
     
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lmdb_connection = lmdb.open(self.lmdbs_path[idx],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[idx])

        image = self.decompress_and_deserialize(lmdb_value)

        if image == None:
            print(self.lmdbs_path[idx])
            #raise InvalidFileException("Invalid file found, skipping")
            return None
            #return image, self.labels[idx]

        return self.transforms(image), self.labels[idx]
        #return read_image(self.images[idx]), self.labels[idx]

class PatchRNADataset(Dataset):
    def __init__(self, patch_data_path, csv_path, img_size, transforms=None,
            max_patches_total=300, quick=False, le=None):
        self.patch_data_path = patch_data_path
        self.csv_path = csv_path
        self.img_size = img_size
        self.transforms = transforms
        self.max_patches_total = max_patches_total
        self.quick = quick
        self.keys = []
        self.images = []
        self.filenames = []
        self.labels = []
        self.lmdbs_path = []
        self.rna_data_arrays = []
        self.le = le
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
            csv_file['patch_data_path'] = [self.patch_data_path] * csv_file.shape[0]
            csv_file['labels'] = [0] * csv_file.shape[0]
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(150)
        
        for i, row in tqdm(csv_file.iterrows()):
            WSI = row['wsi_file_name']
            rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
            rna_data = torch.tensor(rna_data, dtype=torch.float32)
            label = np.asarray(row['Labels'])
            if self.le is not None:
                label = self.le.transform(label.reshape(-1,1))
            label = torch.tensor(label, dtype=torch.int64)
            #label = label.flatten()
            project = row['tcga_project'] 
            if not os.path.exists(os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI)):
                print('Not exist {}'.format(os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI)))
                continue
            
            #try:
            path = os.path.join('/oak/stanford/groups/ogevaert/data/Roche-TCGA/'+project+self.patch_data_path, WSI, WSI)
            try:
                lmdb_connection = lmdb.open(path,
                                            subdir=False, readonly=True, 
                                            lock=False, readahead=False, meminit=False)
            except:
                path = path + '.db'
                try:
                    lmdb_connection = lmdb.open(path,
                                                subdir=False, readonly=True, 
                                                lock=False, readahead=False, meminit=False)
                except:
                    continue
            try:
                with lmdb_connection.begin(write=False) as lmdb_txn:
                    n_patches = lmdb_txn.stat()['entries'] - 1
                    keys = pickle.loads(lz4framed.decompress(lmdb_txn.get(b'__keys__')))
            except Exception as e:
                print(e)
                continue

            n_selected = min(n_patches, self.max_patches_total)
            n_patches= list(range(n_patches))
            n_patches_index = random.sample(n_patches, n_selected)

            for i in n_patches_index:
                #self.images.append(os.path.join(data_path, WSI, WSI + '_patch_{}.png'.format(i)))
                self.images.append(i)
                self.filenames.append(WSI)
                self.labels.append(label)
                self.lmdbs_path.append(path)
                self.keys.append(keys[i])
                self.rna_data_arrays.append(rna_data)
 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lmdb_connection = lmdb.open(self.lmdbs_path[idx],
                                        subdir=False, readonly=True, 
                                        lock=False, readahead=False, meminit=False)
        
        with lmdb_connection.begin(write=False) as txn:
            lmdb_value = txn.get(self.keys[idx])

        image = decompress_and_deserialize(lmdb_value)
        rna_data = self.rna_data_arrays[idx]
        if image == None:
            print(self.lmdbs_path[idx])
            #raise InvalidFileException("Invalid file found, skipping")
            out = {
                'image': image,
                'rna_data': rna_data,
                'labels': self.labels[idx]
            }
        else:
            out = {
                'image': image,
                'rna_data': rna_data,
                'labels': self.labels[idx]
            }
        return out
        #return read_image(self.images[idx]), self.labels[idx]

class RNADataset(Dataset):
    def __init__(self, csv_path, quick=False, num_samples=20, le=None, seed=99):
        self.csv_path = csv_path
        self.data = None
        self.quick = quick
        self.num_samples = num_samples
        self.rna_data = []
        self.vital_status = []
        self.survival_months = []
        self.le = le
        self.labels = []
        self.seed = seed
        self._preprocess()

    def _preprocess(self):
        if type(self.csv_path) == str:
            csv_file = pd.read_csv(self.csv_path)
        else:
            csv_file = self.csv_path
        
        if self.quick:
            csv_file = csv_file.sample(self.num_samples, random_state=self.seed)
        
        rna_columns = [x for x in csv_file.columns if 'rna_' in x]
        if self.le != None:
            self.labels = csv_file['tcga_project'].values
            self.labels = self.le.transform(self.labels.reshape(-1,1))
            self.labels = torch.tensor(self.labels, dtype=torch.int64)
        self.rna_data = csv_file[rna_columns].values.astype(np.float32)
        self.rna_data = torch.tensor(self.rna_data, dtype=torch.float32)
        self.vital_status = csv_file['status'].values.astype(np.float32)
        self.vital_status = torch.tensor(self.vital_status, dtype=torch.float32)
        self.survival_months = csv_file['survival_months'].values.astype(np.float32)
        self.survival_months = torch.tensor(self.survival_months, dtype=torch.float32)

    def __len__(self):
        return len(self.rna_data)
    
    def __getitem__(self, idx):
        return self.rna_data[idx], self.vital_status[idx], self.survival_months[idx]


def normalize_dfs(train_df, val_df, test_df, labels=False, norm_type='standard'):
    def _get_log(x):
        # trick to take into account zeros
        x = np.log(x.replace(0, np.nan))
        return x.replace(np.nan, 0)
    # get list of columns to scale
    rna_columns = [x for x in train_df.columns if 'rna_' in x]
    
    # log transform
    train_df[rna_columns] = train_df[rna_columns].apply(_get_log)
    val_df[rna_columns] = val_df[rna_columns].apply(_get_log)
    test_df[rna_columns] = test_df[rna_columns].apply(_get_log)
    
    '''
    train_df = train_df[rna_columns+rest_columns]
    val_df = val_df[rna_columns+rest_columns]
    test_df = test_df[rna_columns+['wsi_file_name']+['Labels']]
    '''
    rna_values = train_df[rna_columns].values

    if norm_type == 'standard':
        scaler = StandardScaler()
    elif norm_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0,1))
    rna_values = scaler.fit_transform(rna_values)

    train_df[rna_columns] = rna_values
    test_df[rna_columns] = scaler.transform(test_df[rna_columns].values)
    val_df[rna_columns] = scaler.transform(val_df[rna_columns].values)

    return train_df, val_df, test_df, scaler