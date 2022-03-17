import pandas as pd
import numpy as np
import random

from types_ import *

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def create_ssl_csv(csv_path: str, tissue: str):

    data = pd.read_csv(csv_path)

    wsi = data['wsi_file_name'].values
    range_values = list(range(len(wsi)))
    values_wsi = random.sample(range_values, int(0.5*len(wsi)))
    
    pairs = pairwise(values)

    for pair in pairs:
        idx1 = pair[0]
        idx2 = pair[1]
        aux = wsi[idx1]
        wsi[idx1] = wsi[idx2]
        wsi[idx2] = aux

    labels = np.zeros(len(wsi), dtype=np.int64)

    labels[values_wsi] = 1

    data['wsi_file_name'] = wsi
    data['Labels'] = labels

    data.to_csv('GTex_' + str(tissue) + '_data_SSL.csv', index=false)


if __name__ == '__main__':
    create_ssl_csv('../../GTex_Lung_data_noerrors.csv', 'Lung')
