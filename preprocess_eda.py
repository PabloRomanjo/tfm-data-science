import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Cargamos dataset
crc_dataset = pd.read_csv('datasets/dataset_enhancer_crc_aa_c_ml.csv', sep=',')

# Comprobamos los valores que toman las features categóricas para buscar posibles fallos
crc_dataset_cat = crc_dataset.select_dtypes(include=['object'])
for i in crc_dataset_cat.columns:
    uniq = crc_dataset_cat[i].unique()
    print('La feature {} toma los valores: {}'.format(i, str(uniq).strip('[]')))

# Corregimos los valores de la feature gender, en minúscula
crc_dataset['gender'] = crc_dataset['gender'].str.lower()
crc_dataset.loc[crc_dataset['disease'] == 'ADVANCED ADENOMA', 'stage'] = \
    crc_dataset.loc[crc_dataset['disease'] == 'ADVANCED ADENOMA', 'stage'].fillna('AA')

# Comprobamos el cambio
print(crc_dataset['gender'].unique())
crc_dataset = crc_dataset[crc_dataset['disease'] != 'NON-ADVANCED ADENOMA'].reset_index(drop=True)
# Hacemos un hist para comprobar distribución de muestras por género
fig = px.histogram(crc_dataset, x="disease", color="gender", nbins=4,
                   labels={"disease": "Condición", "count": "Frecuencia"},
                   title="Distribución de muestras de acuerdo a fenotipo y genero")
fig.update_layout(
    xaxis_title="Condición",
    yaxis_title="Número de muestras",
    font=dict(
        family="Arial",
        size=18,
        color="Black"
    ))
fig.show()

# Hacemos un hist para comprobar distribución por etapa
fig = px.histogram(crc_dataset, x="stage", color="gender", nbins=4,
                   labels={"stage": "Estadio", "count": "Frecuencia"},
                   title="Distribución de muestras CCR de acuerdo a estadio",
                   category_orders=dict(stage=['AA', 'I', 'II', 'III', 'IV']))
fig.update_layout(
    xaxis_title="Fase",
    yaxis_title="Número de muestras",
    font=dict(
        family="Arial",
        size=18,
        color="Black"
    ))
fig.show()

# Lo mismo pero por etnia
fig = px.histogram(crc_dataset, x="disease", color="ethnicity", nbins=4,
                   labels={"disease": "Condición", "count": "Frecuencia"},
                   title="Distribución de muestras de acuerdo a fenotipo y etnia")
fig.update_layout(
    xaxis_title="Etnia",
    yaxis_title="Número de muestras",
    font=dict(
        family="Arial",
        size=18,
        color="Black"
    ))
fig.show()

# Distribución de edad por condición
fig = make_subplots(rows=2, cols=3)

# histograma CCR
fig.add_trace(go.Histogram(x=crc_dataset[crc_dataset['disease'] == 'COLORECTAL CANCER']['age_at_collection'],
                           nbinsx=10, marker_color='red', name='CCR'), row=1, col=3)
# histograma control
fig.add_trace(go.Histogram(x=crc_dataset[crc_dataset['disease'] == 'CONTROL']['age_at_collection'],
                           nbinsx=10, marker_color='blue', name='Control'), row=1, col=1)
# histograma AA
fig.add_trace(go.Histogram(x=crc_dataset[crc_dataset['disease'] == 'ADVANCED ADENOMA']['age_at_collection'],
                           nbinsx=10, marker_color='pink', name='AA'), row=1, col=2)

fig.update_layout(title='Distribución de muestras por rango de edad y condición', xaxis=dict(title='Edad'),
                  yaxis=dict(title='Frecuencia'), height=800, width=800)
fig.update_layout(
    xaxis_title="Edad",
    yaxis_title="Número de muestras",
    font=dict(
        family="Arial",
        size=18,
        color="Black"
    ))
fig.show()
