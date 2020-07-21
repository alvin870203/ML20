#!/bin/bash
# Homework 2: Classification
# hw2_logistic.sh
echo 'The script name is : '$0''
echo 'Total parameter number is : '$#''
echo 'Your whole parameter is : '$@''
echo 'The 1st parameter (raw training data - train.csv) : '$1''
echo 'The 2nd parameter (raw testing data - test_no_label.csv) : '$2''
echo 'The 3rd parameter (preprocessed training feature - X_train) : '$3''
echo 'The 4th parameter (training lable - Y_train) : '$4''
echo 'The 5th parameter (preprocessed testing feature - X_test) : '$5''
echo 'The 6th parameter (output path - output_{}.csv) : '$6''
echo 'Start to run Python'
python hw2_classification_logistic.py $1 $2 $3 $4 $5 $6
echo 'End of script'
