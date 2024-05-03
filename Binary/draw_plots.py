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

data = pd.read_csv('F:\Thesis\Binary\Carcinoma\EA\selected_clusters.txt')
draw_barplot(data, 'Selected Clusters')