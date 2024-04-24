# open a sheet of an excel file and input the data into a dataframe

import pandas as pd

# read the text file and input the data into a dataframe
df = pd.read_csv('F:\Thesis\Shamima apu\AFFY HuGeneFL probe_gene symbol.txt', sep='\t')

# print the first 5 rows of the dataframe
print(df.head())

#input the arff file from whose data will be extracted
from scipy.io import arff

data = arff.loadarff('F:\Thesis\Shamima apu\Colon Tumor\Colon.arff')
df2 = pd.DataFrame(data[0])

print(df2.head())

print(df2.columns)

# Create a new dataframe with the column names of the arff file and the corresponding gene name from df
df3_data = []

# Iterate through the rows of df2 and df, and if the gene name matches, add the gene symbol to the new dataframe
for i in range(len(df2.columns)):
    gene_name = df2.columns[i].strip()
    matching_rows = df[df['AFFY HuGeneFL probe'].str.strip() == gene_name]

    if not matching_rows.empty:
        gene_symbol = matching_rows.iloc[0]['Gene name']
        df3_data.append({'AFFY gene name': gene_name, 'gene symbol': gene_symbol})
    else:
        df3_data.append({'AFFY gene name': gene_name, 'gene symbol': 'NA'})

df3_data = pd.DataFrame(df3_data)

# write df3 to a csv file
df3_data.to_csv('F:\Thesis\Shamima apu\Colon Tumor\Colon_CNS_gene_symbol.csv', index=False)