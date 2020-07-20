#!/usr/bin/env python
"""Homework 1: Linear Regression"""

# Import Packages
import sys
import pandas as pd
import numpy as np
import math
import csv

# Load 'train.csv'
# train.csv 的資料為 12 個月中，每個月取 20 天，每天 24 小時的資料(每小時資料有 18 個 features)。
data = pd.read_csv(sys.argv[1], encoding = 'big5')  # sys.argv[1] ==> './train.csv'

# Preprocessing
# 取需要的數值部分，將 'RAINFALL' 欄位全部補 0。
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# Extract Features (1)
# 將原始 4320 (= 12 months * 20 days * 18 features) * 24 (hours) 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料
# ，(共有12個Data)。
# Pseudo code
#     Declare a 18-dim vector (Data)
#     for i_th row in training data :
#         Data[i_th row % 18].append(every element in i_th row)
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# Extract Features (2)
# 每 10 小時為一筆資料。
# 每個月會有 480hrs，每 9 小時形成一個 data (x)，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * 18 的 features (一小時
# 18 個 features * 9 小時)。
# 對應的 target (y) 則有 471 * 12 個(第 10 個小時的 PM2.5)。
# Pseudo code
#     Declare train_x for previous 9-hr data, and train_y for 10th-hr pm2.5
#     for i in all the given data :
#         sample every 10 hrs :
#             train_x.append(previous 9-hr data)
#             train)y.append(the value of 10th-hr pm2.5)
#     add a bias term to every data in train_x
x = np.empty([12 * 471, 18 * 9], dtype = float)  # will convert assigned variable to float type
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:  # 每月的最後一天(第20天）的第15個小時，為該月份的最後一筆資料(第471筆 = 19day*24hr_per_day+15hr)
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)  # vector dim: 1 * (18*9) ==> (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value
print(x)
print(y)

# Normalize (1)
mean_x =  np.mean(x,axis = 0)  # (18*9) * 1
std_x = np.std(x, axis = 0)  # (18*9) * 1
for i in range(len(x)):  # len(x) == 12*471
    for j in range(len(x[0])):  # len(x[0]) == 18*9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
print(x)

# Split Training Data Into "train_set" and "validation_set"
# 這部分是針對作業中 report 的第二題、第三題做的簡單示範，以生成比較中用來訓練的 train_set 和不會被放入訓練、只是用來驗證的 validation_set。
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

# Training
# Implement linear regression
# Pseudo code
#     Declare weight vector, initial lr, and # of iteration
#     for i_th iteration :
#         y' = the inner product of train_x and weight vector
#         Loss = y' - train_y
#         gradient = 2*np.dot((train_x).T, L)
#         weight vector -= learning rate * gradient
# Adagrad
# Pseudo code
#     Declare weight vector, initial lr, and # of iteration
#     Declare prev_gra storing gradients in every previous iterations
#     for i_th iteration :
#         y' = the inner product of train_x and weight vector
#         Loss = y' - train_y
#         gradient = 2*np.dot((train_x).T, L)
#         prev_gra += gra**2
#         ada = np.sqrt(prev_gra)
#         weight vector -= learning rate * gradient / ada
# (和上圖不同處: 下面的 code 採用 Root Mean Square Error)
# 因為常數項的存在，所以 dimension (dim) 需要多加一欄；eps 項是避免 adagrad 的分母為 0 而加的極小數值。
# 每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，透過一次次的 iteration (iter_time) 學習。
dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
learning_rate = 100
iter_time = 1000
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
    if (t % 100 == 0):
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim * 1
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
np.save('weight.npy', w)
print(w)

# Testing
# Predict PM2.5
# Pseudo code
#     read test_x.csv file
#     for every 18 rows:
#         test_x.append([1])
#         test_x.append(9-hr data)
#         test_y = np.dot(weight vector, test_x)
# 載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
test_data = pd.read_csv(sys.argv[2], header = None, encoding = 'big5')  # sys.argv[2] ==> './test.csv'
test_data = test_data.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i : 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240,1]), test_x), axis = 1).astype(float)
print(test_x)

# Prediction
# 說明圖同上
# Predict PM2.5
# Pseudo code
#     read test_x.csv file
#     for every 18 rows:
#         test_x.append([1])
#         test_x.append(9-hr data)
#         test_y = np.dot(weight vector, test_x)
# 有了 weight 和測試資料即可預測 target。
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
print(ans_y)

# Save Prediction to CSV File
with open('submit.csv', mode = 'w', newline = '') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)