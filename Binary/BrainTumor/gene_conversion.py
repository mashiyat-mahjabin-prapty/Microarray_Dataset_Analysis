import pandas as pd

# Read the data
data = pd.read_csv('Binary\BrainTumor\\results\max_genes.txt', sep='\t')

print(data.head())

# read the gene symbol file
gene_symbol = pd.read_csv('Binary\BrainTumor\BrainTumor_CNS_gene_symbol.csv')

print(gene_symbol.head())

# take the intersection of the two dataframes
result = pd.merge(data, gene_symbol, on='AFFY gene name')

print(result.head())

# save the result to a file
result.to_csv('Binary\BrainTumor\\results\max_genes.txt', sep='\t', index=False)