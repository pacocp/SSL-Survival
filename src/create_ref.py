import glob

import pandas as pd
from tqdm import tqdm
import numpy as np

#diseases = ['ESCA', 'CESC', 'CHOL', 'GBM', 'KIRP', 'OV', 'PAAD', 'PRAD', 'UCS', 'UVM']
diseases = ['LUAD']
wsi_names = []
labels = []
tcga_project = []
status = []
survival_months = []
gene_info = pd.read_csv('../../Roche-TCGA/gene_expression/pancer_mRNA.txt', sep=' ',
                        low_memory=False)
genes = gene_info.index.values[:-1]
genes_dict = {}
for gene in genes:
    genes_dict[gene] = []
patient_ids_genes = gene_info.columns.values
patient_ids = []
for disease in diseases:
    print('Processing {}'.format(disease))
    wsi_file_names = glob.glob('../../Roche-TCGA/TCGA-'+disease+'/*.svs')
    survival = pd.read_csv(f'../../Roche-TCGA/survival/{disease}.txt', sep='\t')
    survival_patient_id = survival['Patient ID'].values
    for x in tqdm(wsi_file_names):
        name = x.split('/')[-1].replace('.svs', '')
        split_name = name.split('-')
        patient_id = split_name[0] + '-' + split_name[1] + '-' + split_name[2]
        if '-DX' in name:
            if (patient_id in patient_ids_genes) and (patient_id in survival_patient_id):
                wsi_names.append(name)
                patient = survival.loc[survival['Patient ID'] == patient_id]
                status.append(int(patient['OS_STATUS'].values[0].split(':')[0]))
                survival_months.append(float(patient['OS_MONTHS'].values[0]))
                tcga_project.append('TCGA-'+disease)
                patient_ids.append(patient_id)
                column = gene_info[patient_id].values[:-1]
                for value, key in zip(column, genes):
                    genes_dict[key].append(value)

data = pd.DataFrame()
data['wsi_file_name'] = wsi_names
data['tcga_project'] = tcga_project
data['survival_months'] = survival_months
data['status'] = status
genes_data = pd.DataFrame.from_dict(genes_dict)
whole_data = pd.concat([data, genes_data], axis=1)
whole_data.to_csv('LUAD_overall_survival.csv', index=False)