import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import f1_score, make_scorer, recall_score, confusion_matrix, accuracy_score, precision_score, \
    log_loss, precision_recall_curve, roc_auc_score, auc, brier_score_loss, RocCurveDisplay, \
    PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import utils
import models

seed = 23

# Cargamos genes asociados a enhancers
enhancer_genes = pd.read_csv('enhancer_associated_genes.csv', sep=',')
# Cargamos genes relacionados con cancer
cancer_genes = pd.read_csv('genes_cancer_list.csv')
cancer_genes = list(cancer_genes['0'])

# Cargamos dataset completo
dataset = pd.read_csv('datasets/dataset_enhancer_crc_aa_c_ml.csv', sep=',')
# Filtramos aquellos registros no asociados a CCR o control
dataset = dataset[(dataset['disease'] == 'CONTROL') | (dataset['disease'] == 'COLORECTAL CANCER')].reset_index(
    drop=True)
dataset['stage'] = dataset['stage'].fillna(0)
# Codificamos target y guardamos en variable "target"
dataset['disease'] = dataset['disease'].map({'CONTROL': 0, 'COLORECTAL CANCER': 1})
target = dataset['disease']
# Generamos variable feature, dataset sin target
features = dataset.drop(['samples', 'disease', 'stage', 'ethnicity'], axis=1)
# Codificamos género
features['gender'] = features['gender'].str.lower()
features['gender'] = features['gender'].map({'male': 1, 'female': 2})
# Normalizamos edad
features['age_at_collection'] = features['age_at_collection'] / features['age_at_collection'].median()

print("Tamaño de las clases:")
print("Clase 0:", sum(target == 0))
print("Clase 1:", sum(target == 1))

# Dividimos train-test
X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.3, random_state=seed)

# Buscamos los enhancer relacionados con cancer, guardandolos en "cancer_enhancers"
cancer_enhancers = []
for i in enhancer_genes.values:
    genes = i[1].replace("'", '').split(', ')
    enhancer = i[0]
    for j in genes:
        if j in cancer_genes:
            cancer_enhancers.append(enhancer)
            break
        else:
            pass


# Selección de features (t-test, kbest)
def feature_selection(trainx, trainy, method='ttest', k=100):
    global X_train
    if method == 'ttest':
        # Dividimos train en controles y casos
        controles_X_train = trainx[trainy == 0]
        casos_X_train = trainx[trainy == 1]

        # Seleccionamos enhancers con cambio significativo
        dmg = []
        for i in controles_X_train.columns:
            t_stat, p_value = stats.ttest_ind(controles_X_train[i], casos_X_train[i])
            adjusted_p_value = p_value * len(controles_X_train[i])
            if adjusted_p_value < 0.05:
                dmg.append(i)

        dmg_cancer = [i for i in dmg if i in cancer_enhancers]
        trainx_cancer = trainx[['gender', 'age_at_collection'] + dmg_cancer]

        # Buscamos features con alta correlación entre sí y eliminamos una del par
        corr_matrix = trainx_cancer.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        unselected_features = [column for column in upper.columns if any(upper[column] > 0.9)]
        trainx_cancer = trainx_cancer.drop(unselected_features, axis=1)
        trainx_cancer.to_csv('crc_dataset_supervised_xtrain_ttest.csv')
        X_train = pd.read_csv('crc_dataset_supervised_xtrain_ttest.csv', sep=',', index_col=0)
    if method == 'kbest':
        trainx = trainx[['gender', 'age_at_collection'] + cancer_enhancers]
        # Eliminamos features altamente correlacionadas
        corr_matrix = trainx.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        unselected_features = [column for column in upper.columns if any(upper[column] > 0.9)]
        trainx = trainx.drop(unselected_features, axis=1)
        # Seleccionamos k mejores
        selector = SelectKBest(k=k)
        fit = selector.fit(trainx, trainy)
        selected_features_index = selector.get_support()
        selected_features = trainx.columns[selected_features_index]
        trainx = trainx[selected_features]
        trainx.to_csv('crc_dataset_supervised_xtrain_{}kbest.csv'.format(k))
        X_train = pd.read_csv('crc_dataset_supervised_xtrain_{}kbest.csv'.format(k), sep=',', index_col=0)


# Feature selection con t-test
feature_selection(X_train, y_train, method='ttest')
print('Tras este proceso contamos con {} features.'.format(len(X_train.columns)))
# Seleccionamos estas features en x_test
X_test = X_test[X_train.columns]


# Tuning de hiperparámetros
def tune_models(mod, tune=False):
    if tune:
        if mod == 'svm_lin':
            svm = SVC()
            param_grid = [{"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel": ["linear"],
                           "gamma": [0, 1, "scale", "auto"], "random_state": [seed], 'probability': [True]}]
            grid_search = GridSearchCV(svm, param_grid, scoring=utils.specificity_scorer, cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            print(f"Mejores hiperparámetros svm: {grid_search.best_params_}")
            print(f"Puntuación de cross-validation svm: {grid_search.best_score_:.4f}")
            print(f"Puntuación en el conjunto de prueba svm: {grid_search.score(X_test, y_test):.4f}")
            print("-------")
        if mod == 'svm_rbf':
            svm = SVC()
            param_grid = [{"C": [0.0001, 0.001, 0.01, 0.1, 1, 10], "kernel": ["rbf"],
                           "gamma": [0, 1, "scale", "auto"], "random_state": [seed], 'probability': [True]}]
            grid_search = GridSearchCV(svm, param_grid, scoring=utils.specificity_scorer, cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            print(f"Mejores hiperparámetros svm: {grid_search.best_params_}")
            print(f"Puntuación de cross-validation svm: {grid_search.best_score_:.4f}")
            print(f"Puntuación en el conjunto de prueba svm: {grid_search.score(X_test, y_test):.4f}")
            print("-------")
        if mod == 'gb':
            gb = GradientBoostingClassifier()
            param_grid = [{"learning_rate": [0.05, 0.075, 0.1, 0.15, 0.2],
                           "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4, 8, 10],
                           "max_depth": [3, 5, 8], "max_features": ["log2"],
                           "criterion": ["friedman_mse", "squared_error"],
                           "n_estimators": [100, 500, 1000], "random_state": [seed]}]
            grid_search = GridSearchCV(gb, param_grid, scoring=utils.specificity_scorer, cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            print(f"Mejores hiperparámetros gb: {grid_search.best_params_}")
            print(f"Puntuación de cross-validation gb: {grid_search.best_score_:.4f}")
            print(f"Puntuación en el conjunto de prueba gb: {grid_search.score(X_test, y_test):.4f}")
            print("-------")
        if mod == 'rf':
            rf = RandomForestClassifier()
            param_grid = [
                {'n_estimators': [200, 300, 500, 1000], 'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 8],
                 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4, 8, 10],
                 'max_features': ['sqrt', 'log2'], "random_state": [seed]}]
            grid_search = GridSearchCV(rf, param_grid, scoring=utils.specificity_scorer, cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            print(f"Mejores hiperparámetros rf: {grid_search.best_params_}")
            print(f"Puntuación de cross-validation rf: {grid_search.best_score_:.4f}")
            print(f"Puntuación en el conjunto de prueba rf: {grid_search.score(X_test, y_test):.4f}")
            print("-------")
        if mod == 'dt':
            dt = DecisionTreeClassifier()
            param_grid = [{'min_samples_leaf': [1, 2, 4, 8, 10], 'max_depth': range(1, 10),
                           'min_samples_split': range(1, 10), 'criterion': ['gini', 'entropy'], 'random_state': [seed]}]
            grid_search = GridSearchCV(dt, param_grid, scoring="f1", cv=5, verbose=2)
            grid_search.fit(X_train, y_train)
            print(f"Mejores hiperparámetros dt: {grid_search.best_params_}")
            print(f"Puntuación de cross-validation dt: {grid_search.best_score_:.4f}")
            print(f"Puntuación en el conjunto de prueba dt: {grid_search.score(X_test, y_test):.4f}")
            print("-------")

# Ajustamos los hiperparámetros de los modelos
tune_models('svm_lin')
tune_models('svm_rbf')
tune_models('gb')
tune_models('rf')
tune_models('dt')

# Cargamos los modelos t-test con parámetros seleccionados
svm_lin = models.svm_lin
svm_rbf = models.svm_rbf
gb = models.gb
dt = models.dt
rf = models.rf
svc_rbf_ensemble = models.svc_rbf_ensemble
svc_lin_ensemble = models.svc_lin_ensemble

# Evaluamos modelos con validación cruzada
cutoff_df = pd.DataFrame(columns=['Fold', 'Threshold', 'Sensitivity', 'Specificity'])
models = [svm_lin]  # Modificar para probar otros modelos
for model in models:
    model_name = type(model).__name__
    cv = StratifiedKFold(n_splits=5)
    cv_results = {key: [] for key in
                  ["split", "stage", "accuracy", "recall", "f1", "precision", "specificity", "logloss",
                   "auroc", "brier", "auc-pr"]}
    pr_curves = []
    feature_importances = []
    for i1, (train, test) in enumerate(cv.split(X_train, y_train)):

        # Inner train-test split
        X_train_cv, X_test_cv = X_train.iloc[train], X_train.iloc[test]
        y_train_cv, y_test_cv = y_train.iloc[train], y_train.iloc[test]

        # Model fit and predictions
        model.fit(X_train_cv, y_train_cv)
        predictions = model.predict(X_test_cv)
        proba_predictions = model.predict_proba(X_test_cv)[:, 1]
        feature_importances.append(model.coef_[0])

        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # Threshold adjust
        ## Save to plot
        for i in thresholds:
            predictions_i = (proba_predictions >= i).astype('int')
            cm_i = confusion_matrix(y_test_cv, predictions_i)
            total1 = sum(sum(cm_i))
            specificity_i = cm_i[0, 0] / (cm_i[0, 0] + cm_i[0, 1])
            sensitivity_i = cm_i[1, 1] / (cm_i[1, 0] + cm_i[1, 1])
            cutoff_df.loc[len(cutoff_df)] = [i1 + 1, i, sensitivity_i, specificity_i]

        threshold = 0.5
        if threshold != 0.5:
            predictions = (proba_predictions >= threshold).astype('int')

        # By stage
        stages = ['all']
        for s in stages:
            if s != 'all':
                label_predictions = pd.DataFrame({'label': y_test_cv, 'predictions': predictions,
                                                  'proba': proba_predictions})
                s_index = dataset.loc[y_test_cv.index][(dataset['stage'] == s) | (dataset['stage'] == 0)]['stage'].index
                label_predictions = label_predictions.loc[s_index]
                y_test_cv_s = label_predictions['label']
                predictions_s = label_predictions['predictions']
                proba_predictions_s = label_predictions['proba']
            else:
                label_predictions = pd.DataFrame({'label': y_test_cv, 'predictions': predictions,
                                                  'proba': proba_predictions})
                y_test_cv_s = label_predictions['label']
                predictions_s = label_predictions['predictions']
                proba_predictions_s = label_predictions['proba']
            # Metrics
            cv_results["split"].append(i1 + 1)
            cv_results["stage"].append(s)
            cv_results["accuracy"].append(accuracy_score(y_test_cv_s, predictions_s))
            cv_results["recall"].append(recall_score(y_test_cv_s, predictions_s))
            cv_results["f1"].append(f1_score(y_test_cv_s, predictions_s))
            cv_results["precision"].append(precision_score(y_test_cv_s, predictions_s))
            cv_results['specificity'].append(utils.specificity_score(y_test_cv_s, predictions_s))
            cv_results["logloss"].append(log_loss(y_test_cv_s, proba_predictions_s))
            cv_results["brier"].append(brier_score_loss(y_test_cv_s, proba_predictions_s))
            cv_results["auroc"].append(roc_auc_score(y_test_cv_s, proba_predictions_s))
            precision, recall, _ = precision_recall_curve(y_test_cv_s, proba_predictions_s)
            cv_results["auc-pr"].append(auc(recall, precision))
            pr_curves.append((precision, recall, round(auc(recall, precision), 2)))
            # Visualization
            # Hist CV
            df_hist = pd.DataFrame(list(proba_predictions_s), list(y_test_cv_s))
            plt.title('Fold {}'.format(i1 + 1))
            plt.hist(df_hist[df_hist.index == 0][0], bins=10, range=(0, 1), color='blue',
                     alpha=0.5, label='Control')
            plt.hist(df_hist[df_hist.index == 1][0], bins=10, range=(0, 1), color='red',
                     alpha=0.5, label='CCR')
            plt.grid(False)
            plt.legend()
            plt.show()
            # Confusion matrix
            cm = confusion_matrix(y_test_cv_s, predictions_s)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Control', 'CCR'])
            cm_display.plot()
            plt.title('Fold {}'.format(i1 + 1))
            plt.grid(False)
            plt.show()

        # Visualization
        ## PR curve
        fig, ax = plt.subplots(figsize=(6, 6))

        # Iterar sobre las curvas almacenadas en pr_curves y trazarlas en el mismo gráfico
        for i, (p, r, a) in enumerate(pr_curves):
            ax.plot(r, p, label='Fold {} (AUC = {})'.format(i + 1, a))
        ax.set_title("Precision-Recall curve for {}\n(Positive label 'CCR')".format(model_name))
        ax.legend()
        plt.show()

    # Plot recall vs specificity
    mean_sensitivity = []
    mean_specificity = []
    std_deviation_se = []
    std_deviation_sp = []

    # Calcular la sensibilidad y especificidad media y desviación estándar para cada umbral
    for t in thresholds:
        # Filtrar los datos para el umbral actual
        threshold_data = cutoff_df[cutoff_df["Threshold"] == t]

        # Calcular la media de sensibilidad y especificidad
        mean_sensitivity.append(np.mean(threshold_data["Sensitivity"]))
        mean_specificity.append(np.mean(threshold_data["Specificity"]))

        # Calcular la desviación estándar de sensibilidad y especificidad
        std_deviation_se.append(np.std(threshold_data["Sensitivity"]))
        std_deviation_sp.append(np.std(threshold_data["Specificity"]))

    # Convertir las listas a arrays numpy
    mean_sensitivity = np.array(mean_sensitivity)
    mean_specificity = np.array(mean_specificity)
    std_deviation_se = np.array(std_deviation_se)
    std_deviation_sp = np.array(std_deviation_sp)
    # Crear el gráfico
    plt.errorbar(thresholds, mean_sensitivity, yerr=std_deviation_se, label='Sensitivity', fmt='-o')
    plt.errorbar(thresholds, mean_specificity, yerr=std_deviation_sp, label='Specificity', fmt='-o')

    # Configurar etiquetas y leyenda
    plt.xlabel('Threshold')
    plt.ylabel('Mean Value')
    plt.legend()

    # Mostrar el gráfico
    plt.show()

    for s in stages:
        accuracy = [accu for accu, stage in zip(cv_results['accuracy'], cv_results['stage']) if stage == s]
        recall = [accu for accu, stage in zip(cv_results['recall'], cv_results['stage']) if stage == s]
        f1 = [accu for accu, stage in zip(cv_results['f1'], cv_results['stage']) if stage == s]
        precision = [accu for accu, stage in zip(cv_results['precision'], cv_results['stage']) if stage == s]
        specificity = [accu for accu, stage in zip(cv_results['specificity'], cv_results['stage']) if stage == s]
        logloss = [accu for accu, stage in zip(cv_results['logloss'], cv_results['stage']) if stage == s]
        brier = [accu for accu, stage in zip(cv_results['brier'], cv_results['stage']) if stage == s]
        auroc = [accu for accu, stage in zip(cv_results['auroc'], cv_results['stage']) if stage == s]
        aucpr = [accu for accu, stage in zip(cv_results['auc-pr'], cv_results['stage']) if stage == s]

        # print results
        print('Metrics for stage {}'.format(s))
        print('Accuracy: {} (+/- {})'.format(np.mean(accuracy), np.std(accuracy)))
        print('Recall: {} (+/- {})'.format(np.mean(recall), np.std(recall)))
        print('F1: {} (+/- {})'.format(np.mean(f1), np.std(f1)))
        print('Precision: {} (+/- {})'.format(np.mean(precision), np.std(precision)))
        print('Specificity: {} (+/- {})'.format(np.mean(specificity), np.std(specificity)))
        print('Brier: {} (+/- {})'.format(np.mean(brier), np.std(brier)))
        print('Log-loss: {} (+/- {})'.format(np.mean(logloss), np.std(logloss)))
        print('AUC-ROC: {} (+/- {})'.format(np.mean(auroc), np.std(auroc)))
        print('AUC-PR: {} (+/- {})'.format(np.mean(aucpr), np.std(aucpr)))

    # Predict test set
    model.fit(X_train, y_train)
    proba_predictions_test = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    predictions_test = model.predict(X_test)
    if threshold != 0.5:
        predictions_test = (proba_predictions_test >= threshold).astype('int')
    precision_test, recall_test, _ = precision_recall_curve(y_test, proba_predictions_test)
    print('\nMetrics for test set:')
    print('Accuracy: {}'.format(accuracy_score(y_test, predictions_test)))
    print('Recall: {}'.format(recall_score(y_test, predictions_test)))
    print('F1: {}'.format(f1_score(y_test, predictions_test)))
    print('Precision: {}'.format(precision_score(y_test, predictions_test)))
    print('Specificity: {}'.format(utils.specificity_score(y_test, predictions_test)))
    print('Brier: {}'.format(brier_score_loss(y_test, proba_predictions_test)))
    print('Log-loss: {}'.format(log_loss(y_test, proba_predictions_test)))
    print('AUC-ROC: {}'.format(roc_auc_score(y_test, proba_predictions_test)))
    print('AUC-PR: {}'.format(auc(recall_test, precision_test)))
    # Calculate fraction of positive outcomes and average predicted probability in each bin
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, proba_predictions_test, n_bins=10)
    # Plot the calibration curve
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
    plt.grid(True)
    plt.show()
    # Precision Recall Curve in test
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    viz2 = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    _ = viz2.ax_.set_title("Precision-Recall curve for {}\n(Positive label 'CCR')".format(model_name))
    plt.show()
    # Hist test
    df_hist = pd.DataFrame(list(proba_predictions_test), list(y_test))
    plt.hist(df_hist[df_hist.index == 0][0], bins=10, range=(0, 1), color='blue',
             alpha=0.5, label='Control')
    plt.hist(df_hist[df_hist.index == 1][0], bins=10, range=(0, 1), color='red',
             alpha=0.5, label='CCR')


# Visualización de resultados

# roc-auc train
roc_train = False
if roc_train:
    cv = StratifiedKFold(n_splits=5)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    viz = RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name="ROC in test".format(model_name),
        alpha=1,
        lw=1,
        ax=ax,
        color='red'
    )
    for fold, (train, test) in enumerate(cv.split(X_train, y_train)):
        model.fit(X_train.iloc[train], y_train.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            model,
            X_train.iloc[test],
            y_train.iloc[test],
            name=f"ROC fold {fold}",
            alpha=0.2,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label 'CCR')",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()

threshold = False
if threshold:
    from yellowbrick.classifier import DiscriminationThreshold

    visualizer = DiscriminationThreshold(model, quantiles=np.array([0.25, 0.5, 0.75]),
                                         cv=StratifiedKFold(n_splits=5),
                                         exclude="queue_rate")
    visualizer.fit(X_train, y_train)
    plt.grid(True)
    visualizer.show()

# Compute the average feature importance over all folds
plt.close()
avg_feature_importance = np.mean(feature_importances, axis=0)

# Sort the features by importance
sorted_idx = np.argsort(avg_feature_importance)

# Plot the feature importances
sns.barplot(x=X_train.columns[sorted_idx], y=avg_feature_importance[sorted_idx])
plt.xticks(rotation=90, fontsize=8)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
