import pandas as pd

names = pd.read_csv('F:\Thesis\Binary\ColonTumor\\raw data\\names2.csv')

print(names.head())

# take only the first column and split each cell by tab
names = names.iloc[:, 0].str.split('\t')

# names will be only the second element of the split
names = [name[1] for name in names]

print(names[:5])

mapping = pd.read_csv('F:\Thesis\Binary\ColonTumor\\results\max_genes.txt', sep='\t', header=None)

print(mapping.head())

# based on the first column of the mapping file, take that index from the names column and append it to the mapping dataframe in a third column
mapping[2] = [names[int(i)] for i in mapping[0]]

print(mapping.head())

# write the mapping to a file
mapping.to_csv('F:\Thesis\Binary\ColonTumor\\results\max_genes_names.txt', sep='\t', index=False, header=False)