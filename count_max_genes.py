import pandas as pd
import sys 

num_class = sys.argv[1]
folder = sys.argv[2]
# Read the data
data = pd.read_csv(num_class + '/' + folder + '/results/selected_features.csv')

print(data.head())

# each row of data contains gene names got from different runs of the algorithm
# now we want to count the number of times each gene appears in the data
# and keep the gene with the highest count
gene_count = {}
for i in range(data.shape[0]):
    for gene in data.iloc[i, :]:
        if gene in gene_count:
            gene_count[gene] += 1
        else:
            gene_count[gene] = 1

# print(gene_count)

# sort the genes based on their count
sorted_genes = sorted(gene_count.items(), key=lambda x: x[1], reverse=True)

# print(sorted_genes)

# write the genes to a file
with open(num_class + '/' + folder + '/results/max_genes.txt', 'w') as f:
    for gene, count in sorted_genes:
        # print the genes and their count, except the genes with nan values
        if gene == gene:
            f.write(str(gene) + '\t' + str(count) + '\n')

print('Done!')