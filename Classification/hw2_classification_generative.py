#!/usr/bin/env python
"""Homework 2 - Classification by Probabilistic Generative Model"""
# In this section we will discuss a generative approach to binary classification. Again, we will not go through the formulation detailedly. Please
# find Prof. Lee's lecture if you are interested in it.
# 接者我們將實作基於 generative model 的二元分類器，理論細節請參考李宏毅老師的教學影片。

### Import Packages ###
import sys
import numpy as np

### Preparing Data ###
# Training and testing data is loaded and normalized as in logistic regression. However, since LDA is a deterministic algorithm, there is no need to
# build a development set.
# 訓練集與測試集的處理方法跟 logistic regression 一模一樣，然而因為 generative model 有可解析的最佳解，因此不必使用到 development set。

X_train_fpath = sys.argv[3]  # './data/X_train'
Y_train_fpath = sys.argv[4]  # './data/Y_train'
X_test_fpath = sys.argv[5]  # './data/X_test'
output_fpath = sys.argv[6]  # './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes fo the columns that will be normalized. If 'None', all columns will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)
    
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
    
    return X, X_mean, X_std

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _ = _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

data_dim = X_train.shape[1]

### Some Useful Functions ###

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguments:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimention, ]  # 1 row * data_dimention column
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)  # 0 or 1

def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

### Mean and Covariance ###
# In generative model, in-class mean and covariance are needed.
# 在 generative model 中，我們需要分別計算兩個類別內的資料平均與共變異。

# Compute in-class mean
X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)

# Compute in-class covariance
cov_0 = np.zeros((data_dim, data_dim))
cov_1 = np.zeros((data_dim, data_dim))

for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# Sharec covariance is taken as a weighted average of individual in-class covariance.
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

### Computing Weights and Bias
# Directly compute weights and bias from in-class mean and shared variance. Prof. Lee's lecture slides(Classification p.33) gives a concise explanation.
# 權重矩陣與偏差向量可以直接被計算出來，算法可以參考李宏毅老師教學投影片第 33 頁。

# Compute inverse of covariance matrix.
# Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
# Via SVD decomposition, one can get matrix inverse efficiently and accurately.
u, s, v = np.linalg.svd(cov, full_matrices = False)
inv = np.matmul(v.T * 1 / s, u.T)

# Directly compute weights and bias
# model's parameters for P(Y_train==0 | x), so _predict(X, w, b) will return 1 if y==0 for given x.
w = np.dot(inv, mean_0 - mean_1)
b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

# Compute accuracy on training set
Y_train_pred = 1 - _predict(X_train, w, b)  # P(Y_train==1 | x) = 1 - P(Y_train==0 | x)
print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

### Predicting Testing Labels
# Predictions are saved to output_generative.csv.
# 預測測試集的資料標籤並且存在 output_generative.csv 中。

# Predict testing labels
predictions = 1 - _predict(X_test, w, b)
with open(output_fpath.format('generative'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0:10]:
    print(features[i], w[i])