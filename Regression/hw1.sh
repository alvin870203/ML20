#!/bin/bash
# Homework 1: Linear Regression
# hw1.sh
echo 'The script name is : '$0''
echo 'Total parameter number is : '$#''
echo 'Your whole parameter is : '$@''
echo 'The 1st parameter : '$1''
echo 'The 2nd parameter : '$2''
echo 'Start to run Python'
python hw1_regression.py $1 $2
