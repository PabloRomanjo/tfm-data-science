import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
seed = 23

# Models t-test
svm_lin = SVC(C=0.01, gamma=0, kernel='linear', random_state=23, probability=True)
svm_rbf = SVC(C=10, gamma='scale', kernel='rbf', random_state=23, probability=True)
gb = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, max_features='log2', min_samples_leaf=10,
                                min_samples_split=5, n_estimators=1000, criterion='friedman_mse', random_state=seed)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_leaf=1, min_samples_split=6,
                            random_state=seed)
rf = RandomForestClassifier(criterion='gini', max_depth=8, max_features='sqrt', min_samples_leaf=2,
                            min_samples_split=10, n_estimators=300, random_state=seed)
with open('ensemble_model_10svc_rbf.save', 'rb') as file:
    svc_rbf_ensemble = pickle.load(file)
with open('ensemble_model_10svc_lin.save', 'rb') as file:
    svc_lin_ensemble = pickle.load(file)


# Models Kbest 100
svm_lin_k100 = SVC(C=0.01, gamma=0, kernel='linear', random_state=seed, probability=True)
svm_rbf_k100 = SVC(C=10, gamma='scale', kernel='rbf', random_state=seed, probability=True)
gb_k100 = GradientBoostingClassifier(learning_rate=0.15, max_depth=8, max_features='log2', min_samples_leaf=1,
                                     min_samples_split=5, n_estimators=1000, criterion='squared_error',
                                     random_state=seed)
dt_k100 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=2, min_samples_split=8,
                                 random_state=seed)
rf_k100 = RandomForestClassifier(criterion='gini', max_depth=8, max_features='sqrt', min_samples_leaf=2,
                                 min_samples_split=2, n_estimators=200, random_state=seed)

# Models Kbest 200
svm_lin_k200 = SVC(C=0.01, gamma=0, kernel='linear', random_state=seed, probability=True)
svm_rbf_k200 = SVC(C=10, gamma='scale', kernel='rbf', random_state=seed, probability=True)
gb_k200 = GradientBoostingClassifier(learning_rate=0.1, max_depth=8, max_features='log2', min_samples_leaf=1,
                                     min_samples_split=10, n_estimators=500, criterion='squared_error',
                                     random_state=seed)
dt_k200 = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=10, min_samples_split=2,
                                 random_state=seed)
rf_k200 = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_leaf=2,
                                 min_samples_split=10, n_estimators=300, random_state=seed)

# Models Kbest 300
svm_lin_k300 = SVC(C=0.01, gamma=0, kernel='linear', random_state=seed, probability=True)
svm_rbf_k300 = SVC(C=10, gamma='scale', kernel='rbf', random_state=seed, probability=True)
gb_k300 = GradientBoostingClassifier(learning_rate=0.05, max_depth=8, max_features='log2', min_samples_leaf=10,
                                     min_samples_split=5, n_estimators=500, criterion='friedman_mse', random_state=seed)
dt_k300 = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=10, min_samples_split=2,
                                 random_state=seed)
rf_k300 = RandomForestClassifier(criterion='entropy', max_depth=8, max_features='sqrt', min_samples_leaf=2,
                                 min_samples_split=10, n_estimators=300, random_state=seed)