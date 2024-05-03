# this program is for shap analysis of a binary classification model
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
from sklearn.preprocessing import LabelEncoder

def shap_analysis(model, X, y, feature_names, dataset, class_names):
    # create a shap explainer
    explainer = shap.KernelExplainer(model.predict, X)
    # generate shap values
    shap_values = explainer.shap_values(X)

    # generate shap summary plot
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, class_names=class_names, plot_type='bar')

    # save the shap summary plot
    plt.savefig(dataset + '\shap\shap_summary_plot.jpg')

    # generate shap dependence plot for each feature
    for i in range(len(feature_names)):
        # clear previous figure
        plt.clf()

        shap.dependence_plot(i, shap_values, X, feature_names=feature_names, show=False)
        plt.title('Shap Dependence Plot for ' + feature_names[i])
        plt.tight_layout()
        # save the shap dependence plot
        plt.savefig(dataset + '\shap\shap_dependence_plot_' + feature_names[i] + '.jpg')

    # generate shap force plot for each instance
    # Get unique classes
    classes = y['CLASS'].unique()

    # For each class, select two random instances
    for class_label in classes:
        instances = y[y == class_label].sample(2).index
        for instance in instances:
            # Create a new figure for each instance
            plt.figure()
            
            # Generate force plot for the current instance
            shap.force_plot(explainer.expected_value, shap_values[instance], X.iloc[instance], feature_names=feature_names, show=False, matplotlib=True)
            
            plt.title('Shap Force Plot for Class ' + str(class_label) + ' Instance ' + str(instance))
            plt.tight_layout()
            # Save the figure with a unique filename
            plt.savefig(dataset + '\shap\shap_force_plot_' + 'class_' + str(class_label) + '_instance_' + str(instance) + '.jpg')
            
            # Close the current figure to release resources
            plt.close()

    # generate shap decision plot for 2 random instances of each class (total 4 instances)
    # For each class, select two random instances
    for class_label in classes:
        instances = y[y == class_label].sample(2).index
        for instance in instances:
            # Create a new figure for each instance
            plt.figure()

            shap.plots.decision(explainer.expected_value, shap_values[instance], X.iloc[instance], feature_names=feature_names, link='logit', show=False)
            
            plt.title('Shap Decision Plot for Class ' + str(class_label) + ' Instance ' + str(instance))
            plt.tight_layout()

            plt.savefig(dataset + '\shap\shap_decision_plot_' + 'class_' + str(class_label) + '_instance_' + str(instance) + '.jpg')

            # Close the current figure to release resources
            plt.close()

    # generate shap waterfall plot for 2 random instances of each class (total 4 instances)
    # For each class, select two random instances
    for class_label in classes:
        instances = y[y == class_label].sample(2).index
        for instance in instances:
            # Create a new figure for each instance
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[instance], max_display=10, show=False, feature_names=feature_names, features=X.iloc[instance])
            
            plt.title('Shap Waterfall Plot for Class ' + str(class_label) + ' Instance ' + str(instance))
            plt.tight_layout()
            
            plt.savefig(dataset + '\shap\shap_waterfall_plot_' + 'class_' + str(class_label) + '_instance_' + str(instance) + '.jpg')
            # Close the current figure to release resources
            plt.close()

    return

dataset = sys.argv[1]

# load genes with [1] greater than 9
genes = pd.read_csv(dataset + '\\results\max_genes.txt', header=None, sep='\t')
genes = genes[genes[1] > 9][0].tolist()

# load data
data = pd.read_csv(dataset + '\\' + dataset + '.csv', usecols=genes + ['CLASS'])

# split data into X and y
X = data.drop(columns=['CLASS'])
y = data['CLASS']

class_names = y.unique()

le = LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y, columns=['CLASS'])

# read the model
model = joblib.load(dataset + '\\results\\best_model_lr.pkl')
model.fit(X, y.stack())

# shap analysis
shap_analysis(model, X, y, genes, dataset, class_names)