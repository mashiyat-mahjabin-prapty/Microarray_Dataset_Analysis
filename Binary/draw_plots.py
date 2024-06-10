import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# def draw_barplot(data, title):
#     # Sort the data based on the '-log10(q)' column
#     data = data.sort_values(by='-log10(q)', ascending=False)

#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(15, 8))

#     # draw barplot of x='-log10(q)' and y='Enrichment_Term' with horizontal orientation
#     # the coloring should be done according to Source column
#     # there should be a ball on the right side of the barplot that shows the Num_genes with size
#     barplot = sns.barplot(data=data, x='-log10(q)', y='Enrichment_Term', hue='Source', palette='rocket_r', dodge=False)
    
#     def change_width_horizontal(ax, new_value) :
    
#         for patch in ax.patches :
            
#             current_height = patch.get_height()
#             diff = current_height - new_value

#             # we change the bar width
#             patch.set_height(new_value)

#             # we recenter the bar
#             patch.set_y(patch.get_y() + diff * .5)

#     change_width_horizontal(barplot, .4)

#     # Remove the legend created by sns.barplot
#     barplot.legend_.remove()

#     # Iterate through each bar and add a ball on the right side
#     for index, bar in enumerate(barplot.patches):
#         bar_color = bar.get_facecolor()
#         x_val = bar.get_width()
#         y_val = bar.get_y() + bar.get_height() / 2
        
#         # Determine the correct row in the DataFrame
#         num_genes_index = index % data.shape[0]
#         ball_size = data.iloc[num_genes_index]['Num_Genes'] * 20  # Adjust size as needed

#         # Draw the ball with the same color as the bar
#         plt.scatter(x_val, y_val, s=ball_size, color=bar_color, zorder=2)

#         # Draw a line from the end of the bar to the ball for a smooth transition
#         plt.plot([x_val, x_val], [y_val, y_val], color=bar_color, alpha=0.7, lw=2)


#     # Add legend for the ball sizes
#     for size in np.unique(data['Num_Genes']):
#         plt.scatter([], [], s=size*20, color='grey', label=str(size) + ' genes')

#     plt.title(title)
#     plt.tight_layout()
#     plt.legend(title='Number of Genes and Sources', loc='lower right', prop={'size': 14})

#     plt.show()
#     # plt.savefig('F:\Thesis\Binary\Prostate\\barplot_selected_clusters.eps')

import seaborn as sns
import matplotlib.pyplot as plt

def draw_barplot(data, title):
    # Sort the data based on the '-log10(q)' column
    data = data.sort_values(by='-log10(q)', ascending=False)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 8))

    # draw barplot of x='-log10(q)' and y='Enrichment_Term' with horizontal orientation
    # the coloring should be done according to Source column
    # there should be a ball on the right side of the barplot that shows the Num_genes with size
    barplot = sns.barplot(data=data, x='-log10(q)', y='Enrichment_Term', hue='Source', palette='rocket_r', dodge=False)
    
    def change_width_horizontal(ax, new_value) :
    
        for patch in ax.patches :
            
            current_height = patch.get_height()
            diff = current_height - new_value

            # we change the bar width
            patch.set_height(new_value)

            # we recenter the bar
            patch.set_y(patch.get_y() + diff * .5)

    change_width_horizontal(barplot, .35)

    # Remove the legend created by sns.barplot
    barplot.legend_.remove()

    # Iterate through each bar and add a ball on the right side
    for index, bar in enumerate(barplot.patches):
        bar_color = bar.get_facecolor()
        x_val = bar.get_width()
        y_val = bar.get_y() + bar.get_height() / 2
        
        # Determine the correct row in the DataFrame
        num_genes_index = index % data.shape[0]
        ball_size = data.iloc[num_genes_index]['Num_Genes'] * 30  # Adjust size as needed

        # Draw the ball with the same color as the bar
        plt.scatter(x_val, y_val, s=ball_size, color=bar_color, zorder=2)

        # Draw a line from the end of the bar to the ball for a smooth transition
        plt.plot([x_val, x_val], [y_val, y_val], color=bar_color, alpha=0.7, lw=2)


    # Add legend for the ball sizes
    for size in np.unique(data['Num_Genes']):
        plt.scatter([], [], s=size*30, color='grey', label=str(size) + ' genes')

    print(data['Source'].unique())
    handles, labels = barplot.get_legend_handles_labels()
    print(labels)

    leg1 = plt.legend(handles[len(data['Num_Genes'].unique()):], labels[len(data['Num_Genes'].unique()):], title='Source', loc='lower right', prop={'size': 12})
    
    # Add legend for the number of genes
    plt.legend(handles[:len(data['Num_Genes'].unique())], labels[:len(data['Num_Genes'].unique())], title='Number of Genes', loc='center right', prop={'size': 15})
    
    # Add the source legend back to the plot
    plt.gca().add_artist(leg1)
    
    plt.title(title)
    plt.tight_layout()

    plt.savefig('F:\Thesis\Binary\Carcinoma\carcinoma_barplot_selected_clusters.eps')    
    plt.show()



def draw_boxplot(data, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 5))
    
    sns.boxplot(data=data, palette='deep')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('F:\Thesis\Binary\\boxplot_lr_f1.jpg')

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
    plt.savefig('F:\Thesis\Binary\lineplot_Stacking_f1.jpg')

def draw_violinplot(data, title, metric_name):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(13, 5))
    
    sns.violinplot(data=data, palette='bright')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('F:\Thesis\Binary\\violinplot_'+ metric_name + '.eps')

# data = pd.read_csv('F:\Thesis\Binary\Carcinoma\EA\selected_clusters.txt')
# draw_barplot(data, 'Selected Enrichment Clusters')

dataset_names = ['Adenocarcinoma', 'BrainTumor', 'BreastCancer', 'ColonTumor','Gastric', 'Leukemia', 'Lung', 'Lymphoma', 'Myeloma', 'OvarianCancer', 'Prostate']
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
        draw_violinplot(data, 'F1 Score of Logistic Regression on Different Binary Datasets', m)
    elif m == 'F1_Stacking':
        draw_violinplot(data, 'F1 Score of Stacking Classifier on Different Binary Datasets', m)
    else:
        draw_violinplot(data, 'F1 Score of Support Vector Classifier on Different Binary Datasets', m)