import pandas as pd
import numpy as np
import random

from types_ import *

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def create_ssl_csv(csv_path: str):

    data = pd.read_csv(csv_path)
    
    wsi = data['wsi_file_name'].values
    range_values = list(range(len(wsi)))
    values_wsi = random.sample(range_values, int(0.5*len(wsi)))
    
    pairs = pairwise(values_wsi)
    errors = 0
    total = len(range_values)
    for pair in pairs:
        if data.iloc[pair[0]]['tcga_project'] == data.iloc[pair[1]]['tcga_project']:
            values_wsi.remove(pair[0])
            values_wsi.remove(pair[1])
            errors+=1
            continue
        idx1 = pair[0]
        idx2 = pair[1]
        aux = wsi[idx1]
        wsi[idx1] = wsi[idx2]
        wsi[idx2] = aux

    print('{}/{}'.format(errors, total))
    labels = np.zeros(len(wsi), dtype=np.int64)

    labels[values_wsi] = 1

    data['wsi_file_name'] = wsi
    data['Labels'] = labels

    data.to_csv('pancancer_data_SSL.csv', index=False)

if __name__ == '__main__':
    create_ssl_csv('../data/pancancer_overall_survival.csv')
