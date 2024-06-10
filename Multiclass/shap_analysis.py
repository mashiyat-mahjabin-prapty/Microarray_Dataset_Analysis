# this program is for shap analysis of a multiclass classification model
# it will generate a shap summary plot and a shap dependence plot for each feature
# it will also generate a shap force plot for each instance
# it will also generate a shap decision plot for each instance
# it will also generate a shap waterfall plot for each instance
# it will also generate a shap interaction plot for each pair of features

import shap
import pandas as pd
import matplotlib.pyplot as plt
import sys
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

def shap_interaction_plot(expected_values, shap_values, class_names, feature_names, X, dataset):
    # generate shap interaction plot for all pairs of features
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                plt.figure()
                shap_interaction_values = shap_values[i] * shap_values[j]
                shap.summary_plot(shap_interaction_values, X, plot_type='dot', max_display=10, show=False, class_names=class_names)
                plt.tight_layout()
                plt.savefig(dataset + '\shap\shap_interaction_plot_class_' + class_names[i] + '_and_' + class_names[j] + '.eps')
                plt.close()

    return

def shap_summary_plot(expected_values, shap_values, class_names, feature_names, X, dataset):
    plt.figure()
    # generate shap summary plot as bar
    shap.summary_plot(shap_values, X, plot_type='bar', max_display=10, show=False, class_names=class_names)
    # save the shap summary plot
    plt.tight_layout()
    plt.savefig(dataset + '\shap\shap_summary_plot_bar.eps')

    plt.close()

def shap_dependence_plot(expected_values, shap_values, class_name, feature_names, X, dataset):
    # generate shap dependence plot for top 2 features
    top_features = np.abs(shap_values).mean(axis=0).argsort()[::-1][:2]
    # print(top_features)
    # print the name of the top 2 features
    print('Top 2 features:', feature_names[top_features[0]], feature_names[top_features[1]])

    for j in range(2):
        plt.figure()

        # generate shap dependence plot for the current feature
        shap.dependence_plot(top_features[j], shap_values, X, show=False)
        plt.tight_layout()
        # save the shap dependence plot
        plt.savefig(dataset + '\shap\shap_dependence_plot_class_' + class_name + '_' + feature_names[top_features[j]] + '.eps')

        plt.close()

    return

def shap_force_plot(expected_values, shap_values, class_name, feature_names, X, dataset):
    for instance in range(len(X)):
        plt.figure()
        shap.force_plot(expected_values, shap_values[instance], X.iloc[instance], feature_names=feature_names, show=False, matplotlib=True)
        plt.tight_layout()
        plt.savefig(dataset + '\shap\shap_force_plot_class_' + class_name + '_instance_' + str(instance) + '.eps')
        plt.close()

    return

def shap_waterfall_plot(expected_values, shap_values, class_name, feature_names, X, dataset):
    for instance in range(len(X)):
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(expected_values, shap_values[instance], max_display=10, show=False, feature_names=feature_names, features=X.iloc[instance])
        plt.tight_layout()
        plt.savefig(dataset + '\shap\shap_waterfall_plot_class_' + class_name + '_instance_' + str(instance) + '.eps')
        plt.close()

    return

def shap_decision_plot(expected_values, shap_values, class_name, feature_names, X, dataset):
    for instance in range(len(X)):
        plt.figure()
        shap.plots.decision(expected_values, shap_values[instance], X.iloc[instance], feature_names=feature_names, link='logit', show=False)
        plt.tight_layout()
        plt.savefig(dataset + '\shap\shap_decision_plot_class_' + class_name + '_instance_' + str(instance) + '.eps')
        plt.close()

    return
def class_labels(row_index, y_ohe, class_names):
    class_count = len(class_names)
    return [
        f"Class {class_names[i]} ({y_ohe[row_index, i].round(2):.2f})"
        for i in range(class_count)
    ]

def shap_multioutput_decision_plot(expected_values, shap_values, class_names, feature_names, X, y, y_pred, samples, dataset):
    for instance in range(len(samples)):
        plt.figure()
        shap.multioutput_decision_plot(expected_values, shap_values, row_index = samples[instance], highlight = y.loc[samples[instance]], show=False, legend_location='lower right', legend_labels=class_labels(samples[instance], y_pred, class_names), feature_names=feature_names)
        plt.tight_layout()
        plt.savefig(dataset + '\shap\shap_multioutput_decision_plot_instance_' + str(samples[instance]) + '.eps')
        plt.close()
    return


def shap_analysis(model, X_train, y_train, X_test, y_test, feature_names, dataset, class_names):
    # create a shap explainer
    explainer = shap.TreeExplainer(model)
    # generate shap values
    shap_values = explainer.shap_values(X_test)
    classes = y_train['CLASS'].unique()

    # generate shap summary plot
    shap_summary_plot(explainer.expected_value, shap_values, class_names, feature_names, X_test, dataset)

    # for each class generate shap dependence plot
    for i in range(len(class_names)):
        shap_dependence_plot(explainer.expected_value, shap_values[i], class_names[i], feature_names, X_test, dataset)

    # generate shap force plot for 2 random instances of each class
    for class_label in range(len(class_names)):
        instances = y_test[y_test == class_label].sample(1).index

        shap_force_plot(explainer.expected_value[class_label], shap_values[class_label], class_names[class_label], feature_names, X_test.loc[instances], dataset)


    # perform ohe in y for multioutput decision plot
    y_pred = model.predict_proba(X_test)

    for classe in range(len(classes)):
        # pass all the samples of the class
        samples = y_test[y_test['CLASS'] == classes[classe]].index

        print(len(samples))
        # print(samples[0], samples[1])

        shap_multioutput_decision_plot(explainer.expected_value, shap_values, class_names, feature_names, X_test, y_test, y_pred, samples, dataset)
    
        # shap_decision_plot(explainer.expected_value[class_label], shap_values[class_label], class_names[class_label], feature_names, X.loc[instances], dataset)

    # generate shap waterfall plot for 2 random instances of each class
    for class_label in range(len(class_names)):
        instances = y_test[y_test == class_label].sample(1).index
        shap_waterfall_plot(explainer.expected_value[class_label], shap_values[class_label], class_names[class_label], feature_names, X_test.loc[instances], dataset)

    return

dataset = sys.argv[1]

# load genes 
genes = pd.read_csv(dataset + '\\results\max_genes.txt', header=None, sep='\t')
genes = genes[genes[1] > 10][0].tolist()
# make the gene names as string if it is a number
if all(isinstance(gene, (int, float)) for gene in genes):
    genes = [str(int(gene)) for gene in genes]

# load data
data = pd.read_csv(dataset + '\\' + dataset + '.csv', usecols=genes + ['CLASS'])

# split data into X and y
X = data.drop(columns=['CLASS'])
# standardize X
X = (X - X.mean()) / X.std()
X = pd.DataFrame(X, columns=genes)

y = data['CLASS']

class_names = y.unique()
class_names.sort()
if all(isinstance(class_name, (int, float)) for class_name in class_names):
    class_names = [str(int(class_name)) for class_name in class_names]
# print(class_names)

# exit()
le = LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y, columns=['CLASS'])

# print the actual class names mapping
# print(le.inverse_transform([0, 1, 2]))

# read the model
# model = joblib.load(dataset + '\\results\\best_model_lr.pkl')
model = xgb.XGBClassifier()

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = pd.DataFrame(X_train, columns=genes)
X_test = pd.DataFrame(X_test, columns=genes)
y_train = pd.DataFrame(y_train, columns=['CLASS'])
y_test = pd.DataFrame(y_test, columns=['CLASS'])

# reset all indices
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

model.fit(X_train, y_train)

# shap analysis
shap_analysis(model, X_train, y_train, X_test, y_test, genes, dataset, class_names)