import pandas as pd
import sys 

# open a file to write the output
sys.stdout = open('info_about_datasets.txt', 'w')    

class_type = ['Binary', 'Multiclass']
file_name = [['Adenocarcinoma', 'BrainTumor', 'BreastCancer', 'ColonTumor', 'Gastric', 'Leukemia', 'Lung', 'Lymphoma', 'Myeloma', 'OvarianCancer', 'Prostate'],
             ['BrainCancer', 'Crohns', 'EndometrialCancer', 'Glioma', 'Leukemia_3', 'Leukemia_4', 'LungCancer', 'Lymphoma', 'MLL', 'SRBCT']]

for i in range(len(class_type)):
    for j in range(len(file_name[i])):
        df = pd.read_csv(class_type[i] + '/' + file_name[i][j] + '/' + file_name[i][j] + '.csv')
        
        print('Dataset: ' + file_name[i][j])
        print('Number of instances:' + str(df.shape[0]) + '\nNumber of features:' + str(df.shape[1] - 1) + '\nNumber of classes:' + str(len(df['CLASS'].unique())))
        # print the unique classes
        print('Unique classes: ' + str(df['CLASS'].unique()))
        # print the number of instances per class
        print('Number of instances per class:') 
        print(df['CLASS'].value_counts())
        print('\n')

        # print the average number of feature selected by the feature selection algorithms
        # read the csv file \results\selected_features.csv that has 20 rows of features
        # read the length of each row, sum them and then divide by 20
        df = pd.read_csv(class_type[i] + '/' + file_name[i][j] + '/results/selected_features.csv')
        total_features = 0
        for k in range(20):
            # Split the row by commas and filter out empty strings
            row_data = ','.join(df.iloc[k].dropna().astype(str).values)
            features = [f for f in row_data.split(',') if f]
            # print(features)
            total_features += len(features)
        print('Average number of features selected: ' + str(total_features/20))
        print('\n')