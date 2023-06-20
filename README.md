# Evaluación de la capacidad predictiva de modelos de aprendizaje supervisado para la clasificación de pacientes con cáncer colorrectal

## Sobre este repositorio
Este repositorio forma parte del **Trabajo Fin de Máster realizado por Pablo Román-Naranjo Varela** en el Máster en Ciencia de Datos de la UOC, y titulado **"Evaluación de la capacidad predictiva de modelos de aprendizaje supervisado para la clasificación de pacientes con cáncer colorrectal"**.

## Resumen del Trabajo

El **cáncer colorrectal (CCR)** es la segunda causa de muerte por cáncer en todo el mundo, representando un 9,5% de todas las muertes causadas por esta enfermedad. Además de la edad del paciente, existen otros factores que confieren riesgo de desarrollar CCR y, por tanto, deben considerarse para determinar las poblaciones sobre las que se realizan los programas de cribado. La identificación de estos factores de riesgo permitiría un enfoque personalizado y preciso para cada paciente, lo que ayudaría a mejorar la tasa de supervivencia. De esta manera, **el objetivo principal de este trabajo fue la identificación de marcadores de riesgo útiles para la detección temprana del CCR haciendo uso de algoritmos de aprendizaje automático**.

Para ello, se comparó la capacidad predictiva de diferentes modelos de aprendizaje automático supervisado, como gradient boosting, máquinas de vectores de soporte (SVM) o random forest, haciendo uso de un conjunto de datos público sobre los niveles de hidroximetilación en los enhancers de pacientes con CCR, AAR y controles. Además, se evaluó la idoneidad de K-means para la identificación de subgrupos de pacientes con CCR a partir de estos datos.

Los resultados de este trabajo sugieren que el mejor modelo supervisado para diferenciar los pacientes con CCR y controles a partir de datos de hidroximetilación fue un modelo SVM con kernel lineal, cuya sensibilidad para detectar la enfermedad fue del 58% tras fijar la especificidad al 95%, mejorando el modelo presentado en el artículo de donde se extrajeron los datos. Además, los enhancers que regulan la expresión de genes como MYSM1 o SP1, o los que regulan genes que codifican proteínas involucradas en rutas como las rutas del TGF-β y las integrinas, fueron identificados como los más relevantes al realizar la clasificación de las muestras en CCR o control. Por otro lado, el uso de K-means identificó 6 clústeres entre las muestras del set de datos de hidroximetilación. Dos de estos clústeres estaban conformados principalmente por muestras con CCR, no obstante, estos no se asociaban a una etapa de desarrollo concreta, y la diferenciación entre clústeres no fue clara, obteniendo clústeres muy próximos.

De esta manera, podemos concluir que los datos de hidroximetilación fueron útiles para la identificación de biomarcadores del CCR, obteniendo resultados prometedores mediante aprendizaje automático supervisado. No obstante, estos resultados deben ser interpretados como preliminares, necesitándose una validación en una cohorte externa y un análisis molecular de los biomarcadores señalados.    

## Descripciøn de archivos

En este repositorio se encuentran los archivos y referencias necesarias para replicar los resultados descritos en la memoria final del Trabajo de Fin de Máster titulado Evaluación de la capacidad predictiva de modelos de aprendizaje supervisado para la clasificación de pacientes con cáncer colorrectal". Esta memoria se puede encontrar tanto en el repositorio instuticional de la UOC (https://openaccess.uoc.edu/) como en este repositorio (/memoria).

En la raiz de este directorio podemos encontrar:

1. **make_dataset.py**: Este archivo tiene el código necesario para transformar los datos crudos en un set de datos listo para ser utilizado por los algoritmos de aprendizaje automático. Los datos crudos utilizados se deben descargar desde Zenodo (https://zenodo.org/record/5170265) y copiar en /raw_data. El set de datos generado al utilizar este script se puede encontrar en https://zenodo.org/record/8061669.
2. **eda.py**: Script para visualizar los el set de datos generado. Primer contacto con las características de los datos. Las figuras generadas con este script se pueden encontrar en /figures_eda.
3. **supervised_crc_binary.py**: En este archivo se configuran los modelos supervisados para la clasificación de las muestras en CCR y control, de acuerdo a sus valores de hidroximetilación en enhancers. El archivo models.py reune los diferentes modelos entrenados con los hiperparámetros ya ajustados. Las figuras generadas con este script se pueden encontrar en /figures_models.
4. **unsurpervised_crc.py**: En este script se prueba la validez de K-means para separar los pacientes con CCR en diferentes clusters. Las figuras generadas con este script se pueden encontrar en /figures_unsupervised.
