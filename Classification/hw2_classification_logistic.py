#!/usr/bin/env python
"""Homework 2 - Classification by Logistic Regression"""
# Binary classification is one of the most fundamental problem in machine learning. In this tutorial, you are going to build linear binary classifiers
# to predict whether the income of an indivisual exceeds 50,000 or not. We presented a discriminative and a generative approaches, the logistic
# regression(LR) and the linear discriminant anaysis(LDA). You are encouraged to compare the differences between the two, or explore more
# methodologies. Although you can finish this tutorial by simpliy copying and pasting the codes, we strongly recommend you to understand the
# mathematical formulation first to get more insight into the two algorithms. Please find here and here for more detailed information about the
# two algorithms.
# 二元分類是機器學習中最基礎的問題之一，在這份教學中，你將學會如何實作一個線性二元分類器，來根據人們的個人資料，判斷其年收入是否
# 高於 50,000 美元。我們將以兩種方法: logistic regression 與 generative model，來達成以上目的，你可以嘗試了解、分析兩者的設計理念及差
# 別。針對這兩個演算法的理論基礎，可以參考李宏毅老師的教學投影片 logistic regression 與 generative model。
#
# In this section we will introduce logistic regression first. We only present how to implement it here, while mathematical formulation and analysis
# will be omitted. You can find more theoretical detail in Prof. Lee's lecture.
# 首先我們會實作 logistic regression，針對理論細節說明請參考李宏毅老師的教學影片

### Import Packages ###
import sys
import numpy as np
import matplotlib.pyplot as plt

### Preparing Data ###
# Load and normalize data, and then split training data into training set and development set.
# 下載資料，並且對每個屬性做正規化，處理過後再將其切分為訓練集與發展集。

# Dataset
# This dataset is obtained by removing unnecessary attributes and balancing the ratio between positively and negatively labeled data in the
# Census-Income (KDD) Data Set, which can be found in UCI Machine Learning Repository. Only preprocessed and one-hot encoded data (i.e.
# X_train, Y_train and X_test) will be used in this tutorial. Raw data (i.e. train.csv and test.csv) are provided to you in case you are interested in it.
# 這個資料集是由 UCI Machine Learning Repository 的 Census-Income (KDD) Data Set 經過一些處理而得來。為了方便訓練，我們移除了一些不
# 必要的資訊，並且稍微平衡了正負兩種標記的比例。事實上在訓練過程中，只有 X_train、Y_train 和 X_test 這三個經過處理的檔案會被使用到，
# train.csv 和 test.csv 這兩個原始資料檔則可以提供你一些額外的資訊。

np.random.seed(0)  # fix the random seeds
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

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function splits data into training set and development set
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _ = _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of dat: {}'.format(data_dim))

### Some Useful Functions ###
# Some functions that will be repeatedly used when iteratively updating the parameters.
# 這幾個函數可能會在訓練迴圈中被重複使用到。

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

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

### Functions about Gradient and Loss ###
# Please refers to Prof. Lee's lecture slides (Logistic Regression p.12) for the formula of gradient and loss computation.
# 請參考李宏毅老師上課投影片第 12 頁的梯度及損失函數計算公式。

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguments:
    #     y_pred: probabilistic prediction, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross_entropy: cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

### Training ###
# Everything is prepared, let's start training!
#
# Mini-batch gradient descent is used here, in which training data are split into several mini-batches and each batch is fed into the model
# sequentially for losses and gradients computation. Weights and bias are updated on a mini-batch basis.
#
# Once we have gone through the whole training set, the data have to be re-shuffled and mini-batch gradient desent has to be run on it again. We
# repeat such process until max number of iterations is reached.
#
# 我們使用小批次梯度下降法來訓練。訓練資料被分為許多小批次，針對每一個小批次，我們分別計算其梯度以及損失，並根據該批次來更新模型
# 的參數。當一次迴圈完成，也就是整個訓練集的所有小批次都被使用過一次以後，我們將所有訓練資料打散並且重新分成新的小批次，進行下一
# 個迴圈，直到事先設定的迴圈數量達成為止。

# Zero initialization for weights and bias
w = np.zeros((data_dim,))
b = np.zeros((1,))

# Some parameters for training
max_iter = 10
batch_size = 8
learning_rate = 0.2

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# Calculate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the beginning of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size : (idx + 1) * batch_size]
        Y = Y_train[idx * batch_size : (idx + 1) * batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        # Gradient descent update
        # Learining rate decay with time
        w = w - learning_rate / np.sqrt(step) * w_grad
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1

    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)  # same as _predict(X_train, w, b)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)  # same as _predict(X_dev, w, b) 
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

### Plotting Loss and Accuracy Curve

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

### Predictiong Testing Labels ###
# Predictions are saved to output_logistic.csv.
# 預測測試集的資料標籤並且存在 output_logistic.csv 中。

# Predict testing labels
predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# Print out the most significant weights
ind = np.argsort(np.abs(w))[::-1]
with open(X_test_fpath) as f:
    content = f.readline().strip('\n').split(',')
features = np.array(content)
for i in ind[0 : 10]:
    print(features[i], w[i])