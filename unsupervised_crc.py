import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Cargamos dataset
dataset = pd.read_csv('datasets/dataset_enhancer_crc_aa_c_ml.csv', sep=',')

# Cargamos los 400 top features (proveniente de feature importance SVM lineal)
topenhancer = pd.read_csv('top400enhancers.csv', sep=',')
topenhancer_list = list(topenhancer['top400'])

# Cargamos los genes asociados a enhancers
enhancer_genes = pd.read_csv('enhancer_associated_genes.csv', sep=',')

# Seleccionamos CCR, AAR y controles
dataset = dataset[(dataset['disease'] == 'CONTROL') | (dataset['disease'] == 'COLORECTAL CANCER')
                  | (dataset['disease'] == 'ADVANCED ADENOMA')].reset_index(drop=True)

# Seleccionamos features
features = dataset.drop(['samples', 'disease', 'stage', 'ethnicity'], axis=1)

# Codificamos género y normalizamos la edad
features['age_at_collection'] = features['age_at_collection']/features['age_at_collection'].median()
features['gender'] = features['gender'].str.lower()
features['gender'] = features['gender'].map({'male': 1, 'female': 2})

# Seleccionamos top 400 y definimos target
features_enhancers = features[topenhancer_list]
target = dataset['disease']

# Almacenamos otras variables
other = dataset[['samples', 'stage', 'gender', 'ethnicity', 'age_at_collection']]
other['stage'] = np.where((dataset['disease'] == 'CONTROL') & (other['stage'].isna()), 'C', other['stage'])
other['stage'] = np.where((dataset['disease'] == 'ADVANCED ADENOMA') & (other['stage'].isna()), 'AA', other['stage'])

# PCA
pca = PCA(n_components=4)
scores_pca = pca.fit_transform(features_enhancers)

# K-means
n_clusters = 20
cost = []

# Estimamos k optimo con técnica del codo
for i in range(1, n_clusters):
    kmean = KMeans(i, n_init=10)
    kmean.fit(scores_pca)
    cost.append(kmean.inertia_)

plt.plot(cost, 'bx-')
plt.show()

# Plot de silueta
silhouette_scores = []
for k in range(2, n_clusters):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scores_pca)
    labels = kmeans.labels_
    score = silhouette_score(scores_pca, labels)
    silhouette_scores.append(score)
print(silhouette_scores)
plt.plot(range(2, n_clusters), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.show()

scores_pca_df = pd.DataFrame(scores_pca, columns=['PC1', 'PC2', 'PC3', 'PC4'])
scores_pca_df['disease'] = list(target)
kmean = KMeans(6, n_init=10)
kmean.fit(scores_pca)


# Metricas de calidad de clustering
wcss = kmean.inertia_
print('wcss:', wcss)
silhouette_avg = silhouette_score(scores_pca, kmean.labels_)
print('silhouette_avg:', silhouette_avg)
ch_index = calinski_harabasz_score(scores_pca, kmean.labels_)
print('ch_index:', ch_index)

# Guardamos resultado de clustering
labels = kmean.labels_
dataset['labels'] = labels
dataset_save = dataset[topenhancer_list + ['stage', 'samples', 'ethnicity', 'labels']]
dataset_save.to_csv('figures_unsupervised/result_clustering.csv', index=False)


# Visualizamos
resultados = pd.DataFrame({'disease': list(target), 'cluster': labels, 'stage': other['stage'],
                           'gender': features['gender'], 'age': dataset['age_at_collection']})

# Histogramas por cluster
for c in resultados:
    grid = sns.FacetGrid(resultados, col='cluster')
    grid.map(plt.hist, c)
plt.show()

# PCA 3D
fig = px.scatter_3d(scores_pca_df, x='PC1', y='PC2', z='PC3', color=labels.astype(str))
# fig = px.scatter_3d(scores_pca_df, x='PC1', y='PC2', z='PC3', color=list(target))
# fig = px.scatter_3d(scores_pca_df, x='PC1', y='PC2', z='PC3', color=list(other['stage']))
fig.update_traces(marker=dict(size=4))
fig.show()

# Heatmap
scores_pca_df.sort_values(by='disease', inplace=True)
sns.heatmap(scores_pca_df.drop('disease', axis=1), cmap='YlGnBu')
plt.yticks(ticks=range(len(list(target))), labels=list(scores_pca_df['disease']))
plt.show()
