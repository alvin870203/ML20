#!/usr/bin/env python
"""Homework 3 - Convolution Neural Network"""

### Import Packages ###
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

### Fix Random Seeds ###
torch.manual_seed(446)
np.random.seed(446)

### Read Image ###
# 利用 OpenCV (cv2) 讀入照片並存放在 numpy array 中

def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.int64)  # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img, (128, 128))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

# 分別將 training set、validation set、testing set 用 readfile 函式讀進來
workspace_dir = './food-11'
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of testing data = {}".format(len(test_x)))

### Dataset ###
# 在 PyTorch 中，我們可以利用 torch.utils.data 的 Dataset 及 DataLoader 來"包裝" data，使後續的 training 及 testing 更為方便。
# Dataset 需要 overload 兩個函數：__len__ 及 __getitem__
# __len__ 必須要回傳 dataset 的大小，而 __getitem__ 則定義了當程式利用 [ ] 取值時，dataset 應該要怎麼回傳資料。
# 實際上我們並不會直接使用到這兩個函數，但是使用 DataLoader 在 enumerate Dataset 時會使用到，沒有實做的話會在程式運行階段出現 error。

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert a tensor or an ndarray to PIL Image, because transforms is performed on PIL image.
    transforms.RandomHorizontalFlip(),  # 隨機將圖片水平翻轉
    transforms.RandomRotation(15),  # 隨機旋轉圖片
    transforms.ToTensor(),  # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),  # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape H x W x C to a PIL Image while preserving the value range.
    transforms.ToTensor(),
])

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        if y is not None:
            self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

batch_size = 64  # halfed(origin: 128). Because RuntimeError: CUDA out of memory. Tried to allocate 128.00 MiB (GPU 0; 3.95 GiB total capacity; 3.13 GiB already allocated; 60.69 MiB free; 3.14 GiB reserved in total by PyTorch)
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

### Model ###
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64. 128, 128]
            nn.BatchNorm2d(64),  # <--------/
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )
    
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)  # flatten
        return self.fc(out)

### Training ###

# 使用 training set 訓練，並使用 validation set 尋找好的參數

model =  Classifier().cuda()
loss = nn.CrossEntropyLoss()  # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 60  # doubled(origin: 30), because batch_size doubled

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda())  # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda())  # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward()  # 利用 back propagation 算出每個參數的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # 將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time() - epoch_start_time, \
                train_acc / train_set.__len__(), train_loss / train_set.__len__(), val_acc / val_set.__len__(), val_loss / val_set.__len__()))