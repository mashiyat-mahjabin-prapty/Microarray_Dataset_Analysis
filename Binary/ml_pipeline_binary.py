# dimensionality reduction using RFECV
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import RFECV
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
# do a ranking with XGBoost and take the top 200 features
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import os
import joblib
import sys

# take the name of the folder and dataset from the command line
folder_name = sys.argv[1]

accuracies_lr = []
baccuracies_lr = []
precisions_lr = []
recalls_lr = []
f1_scores_lr = []
roc_aucs_lr = []
mccs_lr = []

accuracies_svm = []
baccuracies_svm = []
precisions_svm = []
recalls_svm = []
f1_scores_svm = []
roc_aucs_svm = []
mccs_svm = []

accuracies_rf = []
baccuracies_rf = []
precisions_rf = []
recalls_rf = []
f1_scores_rf = []
roc_aucs_rf = []
mccs_rf = []

accuracies_voting = []
baccuracies_voting = []
precisions_voting = []
recalls_voting = []
f1_scores_voting = []
roc_aucs_voting = []
mccs_voting = []

accuracies_stacking = []
baccuracies_stacking = []
precisions_stacking = []
recalls_stacking = []
f1_scores_stacking = []
roc_aucs_stacking = []
mccs_stacking = []

# let selected features be a matrix
selected_features = []
# read data
df = pd.read_csv(folder_name + '/' + folder_name + '.csv')

# redirect the results to the result folder inside the folder_name
os.chdir(folder_name + '/results')

le = LabelEncoder()
df['CLASS'] = le.fit_transform(df['CLASS'])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# standardize the data
scaler = StandardScaler()
std_X = scaler.fit_transform(X)
X = pd.DataFrame(std_X, columns=X.columns)

model = XGBClassifier()
model.fit(X, y)
plot_importance(model, max_num_features=20, height=0.5)
plt.rcParams["figure.figsize"] = (100,100)
# plt.show()
plt.savefig('feature_importance.png', bbox_inches='tight')

# get the feature importance
importance = model.feature_importances_
importance = pd.DataFrame(importance, index=X.columns, columns=["Importance"])
importance = importance.sort_values(by=["Importance"], ascending=False)
importance = importance.head(int(X.shape[1]*0.1))

# get the top 10% features
X = X[importance.index]

np.random.seed(42)
random_states = [np.random.randint(1000) for i in range(20)]

best_f1_score = -1
best_model = None

for i in random_states:
    cv = StratifiedKFold(10, shuffle=True, random_state=i)
    # feature extraction
    model = LogisticRegression()
    
    rfecv = RFECV(estimator=model, cv=cv, scoring='f1', n_jobs=-1)
    fit = rfecv.fit(X, y)

    results = open('results_' + str(i) + '.txt', 'w')
    results.write('Optimal number of features: %d' % fit.n_features_)
    results.write('\n')
    # results.write('Selected features: %s' % fit.support_)
    # results.write('\n')

    mask = fit.support_

    X_new = X.loc[:, mask]
    # write the selected feature names to results
    selected_features.append(X.columns[fit.support_])
    # selected_features = X.columns[fit.support_]
    results.write('Selected features: %s' % X.columns[fit.support_].tolist())
    results.write('\n')

    clf1 = LogisticRegression()
    clf2 = SVC(probability=True)
    clf3 = RandomForestClassifier()

    voting = VotingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3)], voting='soft', n_jobs=-1)
    stacking = StackingClassifier(estimators=[('lr', clf1), ('svc', clf2), ('rf', clf3)], final_estimator=LogisticRegression(), n_jobs=-1)


    scoring = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'matthews_corrcoef']
    for clf, label in zip([clf1, clf2, clf3, voting, stacking], ['Logistic Regression', 'SVM', 'Random Forest', 'Voting', 'Stacking']):
        results.write('-----------------------------------')
        results.write('\n')
        results.write(label)
        results.write('\n')
        scores = cross_validate(clf, X_new, y, cv=cv, scoring=scoring)
        results.write('Accuracy: %0.2f (+/- %0.2f)' % (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))
        results.write('\n')
        results.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std() * 2))
        results.write('\n')
        results.write('Precision: %0.2f (+/- %0.2f)' % (scores['test_precision'].mean(), scores['test_precision'].std() * 2))
        results.write('\n')
        results.write('Recall: %0.2f (+/- %0.2f)' % (scores['test_recall'].mean(), scores['test_recall'].std() * 2))
        results.write('\n')
        results.write('F1: %0.2f (+/- %0.2f)' % (scores['test_f1'].mean(), scores['test_f1'].std() * 2))
        results.write('\n')
        results.write('ROC AUC: %0.2f (+/- %0.2f)' % (scores['test_roc_auc'].mean(), scores['test_roc_auc'].std() * 2))
        results.write('\n')
        results.write('MCC: %0.2f (+/- %0.2f)' % (scores['test_matthews_corrcoef'].mean(), scores['test_matthews_corrcoef'].std() * 2))
        results.write('\n')
        results.write('-----------------------------------')
        results.write('\n')
        if label == 'Logistic Regression':
            accuracies_lr.append(scores['test_accuracy'])
            baccuracies_lr.append(scores['test_balanced_accuracy'])
            precisions_lr.append(scores['test_precision'])
            recalls_lr.append(scores['test_recall'])
            f1_scores_lr.append(scores['test_f1'])
            roc_aucs_lr.append(scores['test_roc_auc'])
            mccs_lr.append(scores['test_matthews_corrcoef'])

            if scores['test_f1'].mean() > best_f1_score:
                best_f1_score = scores['test_f1'].mean()
                best_model = clf
        
        elif label == 'SVM':
            accuracies_svm.append(scores['test_accuracy'])
            baccuracies_svm.append(scores['test_balanced_accuracy'])
            precisions_svm.append(scores['test_precision'])
            recalls_svm.append(scores['test_recall'])
            f1_scores_svm.append(scores['test_f1'])
            roc_aucs_svm.append(scores['test_roc_auc'])
            mccs_svm.append(scores['test_matthews_corrcoef'])
        elif label == 'Random Forest':
            accuracies_rf.append(scores['test_accuracy'])
            baccuracies_rf.append(scores['test_balanced_accuracy'])
            precisions_rf.append(scores['test_precision'])
            recalls_rf.append(scores['test_recall'])
            f1_scores_rf.append(scores['test_f1'])
            roc_aucs_rf.append(scores['test_roc_auc'])
            mccs_rf.append(scores['test_matthews_corrcoef'])
        elif label == 'Voting':
            accuracies_voting.append(scores['test_accuracy'])
            baccuracies_voting.append(scores['test_balanced_accuracy'])
            precisions_voting.append(scores['test_precision'])
            recalls_voting.append(scores['test_recall'])
            f1_scores_voting.append(scores['test_f1'])
            roc_aucs_voting.append(scores['test_roc_auc'])
            mccs_voting.append(scores['test_matthews_corrcoef'])
        elif label == 'Stacking':
            accuracies_stacking.append(scores['test_accuracy'])
            baccuracies_stacking.append(scores['test_balanced_accuracy'])
            precisions_stacking.append(scores['test_precision'])
            recalls_stacking.append(scores['test_recall'])
            f1_scores_stacking.append(scores['test_f1'])
            roc_aucs_stacking.append(scores['test_roc_auc'])
            mccs_stacking.append(scores['test_matthews_corrcoef'])

    results.close()

joblib.dump(best_model, 'best_model_lr.pkl')

outfile = open('results.txt', 'w')
# write the random states
outfile.write('Random States: %s' % random_states)
outfile.write('\n')
outfile.write('Logistic Regression')
outfile.write('\n')
outfile.write('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_lr), np.std(accuracies_lr) * 2))
outfile.write('\n')
outfile.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_lr), np.std(accuracies_lr) * 2))
outfile.write('\n')
outfile.write('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions_lr), np.std(precisions_lr) * 2))
outfile.write('\n')
outfile.write('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls_lr), np.std(recalls_lr) * 2))
outfile.write('\n')
outfile.write('F1: %0.2f (+/- %0.2f)' % (np.mean(f1_scores_lr), np.std(f1_scores_lr) * 2))
outfile.write('\n')
outfile.write('ROC AUC: %0.2f (+/- %0.2f)' % (np.mean(roc_aucs_lr), np.std(roc_aucs_lr) * 2))
outfile.write('\n')
outfile.write('MCC: %0.2f (+/- %0.2f)' % (np.mean(mccs_lr), np.std(mccs_lr) * 2))
outfile.write('\n')
outfile.write('--------------------------------------------------------------\n')
outfile.write('--------------------------------------------------------------')
outfile.write('\n')
outfile.write('SVM')
outfile.write('\n')
outfile.write('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_svm), np.std(accuracies_svm) * 2))
outfile.write('\n')
outfile.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_svm), np.std(accuracies_svm) * 2))
outfile.write('\n')
outfile.write('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions_svm), np.std(precisions_svm) * 2))
outfile.write('\n')
outfile.write('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls_svm), np.std(recalls_svm) * 2))
outfile.write('\n')
outfile.write('F1: %0.2f (+/- %0.2f)' % (np.mean(f1_scores_svm), np.std(f1_scores_svm) * 2))
outfile.write('\n')
outfile.write('ROC AUC: %0.2f (+/- %0.2f)' % (np.mean(roc_aucs_svm), np.std(roc_aucs_svm) * 2))
outfile.write('\n')
outfile.write('MCC: %0.2f (+/- %0.2f)' % (np.mean(mccs_svm), np.std(mccs_svm) * 2))
outfile.write('\n')
outfile.write('--------------------------------------------------------------\n')
outfile.write('--------------------------------------------------------------')
outfile.write('\n')
outfile.write('Random Forest')
outfile.write('\n')
outfile.write('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_rf), np.std(accuracies_rf) * 2))
outfile.write('\n')
outfile.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_rf), np.std(accuracies_rf) * 2))
outfile.write('\n')
outfile.write('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions_rf), np.std(precisions_rf) * 2))
outfile.write('\n')
outfile.write('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls_rf), np.std(recalls_rf) * 2))
outfile.write('\n')
outfile.write('F1: %0.2f (+/- %0.2f)' % (np.mean(f1_scores_rf), np.std(f1_scores_rf) * 2))
outfile.write('\n')
outfile.write('ROC AUC: %0.2f (+/- %0.2f)' % (np.mean(roc_aucs_rf), np.std(roc_aucs_rf) * 2))
outfile.write('\n')
outfile.write('MCC: %0.2f (+/- %0.2f)' % (np.mean(mccs_rf), np.std(mccs_rf) * 2))
outfile.write('\n')
outfile.write('--------------------------------------------------------------\n')
outfile.write('--------------------------------------------------------------')
outfile.write('\n')
outfile.write('Voting')
outfile.write('\n')
outfile.write('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_voting), np.std(accuracies_voting) * 2))
outfile.write('\n')
outfile.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_voting), np.std(accuracies_voting) * 2))
outfile.write('\n')
outfile.write('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions_voting), np.std(precisions_voting) * 2))
outfile.write('\n')
outfile.write('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls_voting), np.std(recalls_voting) * 2))
outfile.write('\n')
outfile.write('F1: %0.2f (+/- %0.2f)' % (np.mean(f1_scores_voting), np.std(f1_scores_voting) * 2))
outfile.write('\n')
outfile.write('ROC AUC: %0.2f (+/- %0.2f)' % (np.mean(roc_aucs_voting), np.std(roc_aucs_voting) * 2))
outfile.write('\n')
outfile.write('MCC: %0.2f (+/- %0.2f)' % (np.mean(mccs_voting), np.std(mccs_voting) * 2))
outfile.write('\n')
outfile.write('--------------------------------------------------------------\n')
outfile.write('--------------------------------------------------------------')
outfile.write('\n')
outfile.write('Stacking')
outfile.write('\n')
outfile.write('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_stacking), np.std(accuracies_stacking) * 2))
outfile.write('\n')
outfile.write('Balanced Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies_stacking), np.std(accuracies_stacking) * 2))
outfile.write('\n')
outfile.write('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions_stacking), np.std(precisions_stacking) * 2))
outfile.write('\n')
outfile.write('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls_stacking), np.std(recalls_stacking) * 2))
outfile.write('\n')
outfile.write('F1: %0.2f (+/- %0.2f)' % (np.mean(f1_scores_stacking), np.std(f1_scores_stacking) * 2))
outfile.write('\n')
outfile.write('ROC AUC: %0.2f (+/- %0.2f)' % (np.mean(roc_aucs_stacking), np.std(roc_aucs_stacking) * 2))
outfile.write('\n')
outfile.write('MCC: %0.2f (+/- %0.2f)' % (np.mean(mccs_stacking), np.std(mccs_stacking) * 2))
outfile.write('\n')
outfile.write('--------------------------------------------------------------\n')
outfile.write('--------------------------------------------------------------')
outfile.write('\n')
outfile.close()

# write down the selected features to a csv file
selected_features = pd.DataFrame(selected_features)
selected_features.to_csv('selected_features.csv', index=False)

# write down the metric arrays to a csv file
metrics = pd.DataFrame()
# write the accuracies in a single sheet
metrics['Accuracy_LR'] = accuracies_lr
metrics['Accuracy_SVM'] = accuracies_svm
metrics['Accuracy_RF'] = accuracies_rf
metrics['Accuracy_Voting'] = accuracies_voting
metrics['Accuracy_Stacking'] = accuracies_stacking

metrics['Balanced_Accuracy_LR'] = baccuracies_lr
metrics['Balanced_Accuracy_SVM'] = baccuracies_svm
metrics['Balanced_Accuracy_RF'] = baccuracies_rf
metrics['Balanced_Accuracy_Voting'] = baccuracies_voting
metrics['Balanced_Accuracy_Stacking'] = baccuracies_stacking

metrics['Precision_LR'] = precisions_lr
metrics['Precision_SVM'] = precisions_svm
metrics['Precision_RF'] = precisions_rf
metrics['Precision_Voting'] = precisions_voting
metrics['Precision_Stacking'] = precisions_stacking

metrics['Recall_LR'] = recalls_lr
metrics['Recall_SVM'] = recalls_svm
metrics['Recall_RF'] = recalls_rf
metrics['Recall_Voting'] = recalls_voting
metrics['Recall_Stacking'] = recalls_stacking

metrics['F1_LR'] = f1_scores_lr
metrics['F1_SVM'] = f1_scores_svm
metrics['F1_RF'] = f1_scores_rf
metrics['F1_Voting'] = f1_scores_voting
metrics['F1_Stacking'] = f1_scores_stacking

metrics['ROC_AUC_LR'] = roc_aucs_lr
metrics['ROC_AUC_SVM'] = roc_aucs_svm
metrics['ROC_AUC_RF'] = roc_aucs_rf
metrics['ROC_AUC_Voting'] = roc_aucs_voting
metrics['ROC_AUC_Stacking'] = roc_aucs_stacking

metrics['MCC_LR'] = mccs_lr
metrics['MCC_SVM'] = mccs_svm
metrics['MCC_RF'] = mccs_rf
metrics['MCC_Voting'] = mccs_voting
metrics['MCC_Stacking'] = mccs_stacking

metrics.to_csv('metrics.csv', index=False)