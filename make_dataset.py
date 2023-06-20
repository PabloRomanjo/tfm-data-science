import pandas as pd
import re

# Cargamos raw datasets
train_df = pd.read_csv('raw_data/CRC_enhancer_znorm_training_matrix_v1.csv', sep=',')
validation_df = pd.read_csv('raw_data/CRC_enhancer_znorm_validation_matrix_v1.csv', sep=',')


# Extraer genes asociados a enhancers
genes = []
enhancers = []
for i in list(train_df['feature_name']):
    enhancer_i = re.findall(pattern=r"genehancer_id=([\w-]+)", string=i)
    enhancers.append(enhancer_i[0])
    associated_genes_i = re.findall(pattern=r"connected_gene=([\w-]+)", string=i)
    genes.append(str(associated_genes_i).strip('[]'))
pd.DataFrame({'enhancers': enhancers, "genes": genes}).to_csv("enhancer_associated_genes.csv", index=False)


# Procesar dataset de genes asociados a cancer

genes_cancer = pd.read_csv('dysregulated_genes_cancer.csv', sep=',', header=None)

all_genes = []
for i in genes_cancer[0]:
    genes = i.split(",")[2]
    genes = genes.split(";")
    genes = [x for x in genes if x != '']
    all_genes = all_genes + genes

pd.DataFrame(set(all_genes)).to_csv("genes_cancer_list.csv", index=False)


# Generar dataset desde raw data

# Modificamos el nombre de las futuras features
# Conjunto train
feature_name = [i.split(';') for i in train_df['feature_name']]
feature_name_train = [i[0].split('=')[1] for i in feature_name]
train_df['feature_name'] = feature_name_train
# Conjunto validation
feature_name = [i.split(';') for i in validation_df['feature_name']]
feature_name_validation = [i[0].split('=')[1] for i in feature_name]
validation_df['feature_name'] = feature_name_validation
# Encontramos las features en ambos conjuntos
common_features = [i for i in feature_name_train if i in feature_name_validation]
# Seleccionamos registros con esas features
train_df = train_df[train_df['feature_name'].isin(common_features)].set_index('feature_name')
validation_df = validation_df[validation_df['feature_name'].isin(common_features)].set_index('feature_name')
# Transponemos y convertimos index en columna 'sample'
train_df = train_df.T
train_df['samples'] = train_df.index
validation_df = validation_df.T
validation_df['samples'] = validation_df.index
# Concatenamos los dos dataframes
dataset = pd.concat([train_df.reset_index(drop=True), validation_df.reset_index(drop=True)], axis=0).reset_index(
    drop=True)
# Añadimos info (enfermedad, stage, gender, ethnicity, age) de las muestras
info_samples = pd.read_csv('raw_data/FORESEE_sample_description.tsv', sep='\t', usecols=[
    'sample_name', 'characteristics_indication', 'characteristics_stage', 'characteristics_gender',
    'characteristics_ethnicity', 'characteristics_age_at_collection'])
info_samples.columns = ['samples', 'disease', 'stage', 'gender', 'ethnicity', 'age_at_collection']
dataset = pd.merge(dataset, info_samples, on='samples')

# Guardamos el dataset con todas las muestras
dataset = dataset.reindex(columns=list(info_samples.columns) + [col for col in dataset.columns if
                                                                col not in info_samples.columns])
dataset_crc = dataset[(dataset['disease'] == "COLORECTAL CANCER") | (dataset['disease'] == "CONTROL") |
                      (dataset['disease'] == "ADVANCED ADENOMA") | (dataset['disease'] == "NON-ADVANCED ADENOMA")]

# A su vez guardamos un dataset eliminando muestras non-advanced adenoma o muestras menores a 50 (ml-ready)
dataset_crc_ml = dataset_crc[(dataset_crc['disease'] == "COLORECTAL CANCER") | (dataset_crc['disease'] == "CONTROL") |
                             (dataset_crc['disease'] == "ADVANCED ADENOMA")]

dataset_crc_ml = dataset_crc_ml[dataset_crc_ml['age_at_collection'] >= 50].reset_index(drop=True)
dataset_crc_ml.to_csv('datasets/dataset_enhancer_crc_aa_c_ml.csv', index=False)

