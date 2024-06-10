import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_barplot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    sns.barplot(x='-log10(q)', y='Enrichment_Term', data=data, orient='h', palette=list(reversed(sns.color_palette("YlOrBr", 10))))
    plt.title(title)
    plt.tight_layout()
    plt.show()

def draw_boxplot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    sns.boxplot(data=data, palette='YlOrBr')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def draw_heatmap(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    sns.heatmap(data, annot=True, fmt=".2f", cmap='YlOrBr')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def draw_lineplot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    
    sns.lineplot(data=data, palette='bright')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('F:\Thesis\Multiclass\lineplot_Stacking_f1.jpg')

def draw_violinplot(data, title, metric_name):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 5))
    
    sns.violinplot(data=data, palette='bright')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('F:\Thesis\Multiclass\\violinplot_'+ metric_name + '.eps')

# data = pd.read_csv('F:\Thesis\Binary\Carcinoma\EA\selected_clusters.txt')
# draw_barplot(data, 'Selected Clusters')

dataset_names = ['BrainCancer', 'Crohns', 'EndometrialCancer', 'Glioma', 'Leukemia_3', 'Leukemia_4', 'LungCancer', 'Lymphoma', 'MLL', 'SRBCT']
metric = ['F1_LR', 'F1_Stacking', 'F1_SVM']

for m in metric:
    data = pd.DataFrame(columns = dataset_names)
    for dataset in dataset_names:
        # read only the 'Accuracy_LR' column
        d = pd.read_csv(dataset + '/results/metrics.csv', usecols=[m])
        # for each row, first strip the leading and trailing brackets and then split the string by whitespace and finally turn them as float
        data[dataset] = d[m].str.strip('[]').str.split().apply(lambda x: float(x[0]))

    print(data)
    if m == 'F1_LR':
        draw_violinplot(data, 'F1 Score of Logistic Regression on Different Multiclass Datasets', m)
    elif m == 'F1_Stacking':
        draw_violinplot(data, 'F1 Score of Stacking Classifier on Different Multiclass Datasets', m)
    else:
        draw_violinplot(data, 'F1 Score of Support Vector Classifier on Different Multiclass Datasets', m)