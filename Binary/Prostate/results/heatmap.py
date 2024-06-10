import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def heatmap(df, title):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt=".2f", cmap='coolwarm', cbar=False)
    plt.title(title)
    plt.show()

# Load the data
df = pd.read_csv('selected_features.csv')

# Plot the heatmap
heatmap(df, 'Heatmap of Selected Features')