# open a sheet of an excel file and input the data into a dataframe

import pandas as pd

data = pd.read_csv('F:\Thesis\Binary\ColonTumor\\raw data\data.txt', sep = ' ', header=None)
print(data.head())

data = data.T
print(data.head())

# for the last column replace negative numbers with 1 and positive numbers with 0
for i in range(len(data[2000])):
    if data[2000][i] < 0:
        data[2000][i] = 1
    else:
        data[2000][i] = 0

print(data.head())

# find any missing values
print(data.isnull().sum())

data.to_csv('F:\Thesis\Binary\ColonTumor\\ColonTumor.csv', index=False)