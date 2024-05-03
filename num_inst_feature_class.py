import pandas as pd
import sys 

class_type = ['Binary', 'Multiclass']
file_name = [['Adenoma', 'BrainTumor', 'BreastCancerRelapse', 'ColonTumor', 'Leukemia', 'Lung', 'Lymphoma', 'OvarianCancer', 'Prostate'],
             ['BrainCancer', 'Crohns', 'EndometrialCancer', 'Glioma', 'Leukemia_3', 'Leukemia_4', 'LungCancer', 'Lymphoma', 'MLL', 'SRBCT']]

for i in range(len(class_type)):
    for j in range(len(file_name[i])):
        df = pd.read_csv(class_type[i] + '/' + file_name[i][j] + '/' + file_name[i][j] + '.csv')
        
        print('Dataset: ' + file_name[i][j])
        print('Number of instances:' + str(df.shape[0]) + '\nNumber of features:' + str(df.shape[1] - 1) + '\nNumber of classes:' + str(len(df['CLASS'].unique())))
        # print the unique classes
        print('Unique classes: ' + str(df['CLASS'].unique()))