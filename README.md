This is my undergraduate thesis on microarray gene expression datasets of cancer classification under Professor Dr Mohammad Saifur Rahman.
I have used a pipeline with two steps of feature selections and three different types of evaluation of the selected feature set.

There are 11 binary datasets and 10 multi-class datasets in this repository used in this study. 
For binary and multiclass datasets there is a py file inside each folder with the same pipeline but with some minor coding differences.
Each folder also contains a shap_analysis python file as well as a diagram plotting python file.

In each dataset folder, I have added the metrics we got from the pipeline, selected feature set in each iteration and the best logistic regression model on the basis of F1-score, along with the dataset.
