# -*-coding:utf-8 -*-
'''
@File    :   NN_SAG_covtype.py
@Time    :   2022/05/22 11:10:06
@Author  :   Liang Wang
@Contact :   wangliang.leon20@gmail.com
@Desc    :   神经网络 + SAG优化算法 + covtype_binary数据集
'''
import numpy as np
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, IterableDataset, DataLoader
import os

def get_data(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x

class LibsvmIterDataset(IterableDataset):
    def __init__(self, file_path, n_features):
        """
        file_path: Libsvm格式数据文件地址
        n_features: 特征数
        """
        self.file_path = file_path
        self.n_features = n_features + 1  # augment feature

    def process_line(self, line):
        line = line.strip().split(' ')
        label, values = int(line[0]), line[1:]
        value = np.zeros((self.n_features))
        for item in values:
            idx, val = item.split(':')
            value[int(idx) - 1] = float(val)
        value[-1] = 1.0
        return label, value

    def __iter__(self):
        with open(self.file_path, 'r') as fp:
            for line in fp:
                yield self.process_line(line.strip("\n"))

class LibsvmDataset(Dataset):
    def __init__(self, file_path, n_features):
        """
        file_path: Libsvm格式数据文件地址
        n_features: 特征数，从1开始
        """
        self.file_path = file_path
        self.n_features = n_features + 1

    def process_line(self, line):
        line = line.strip().split(' ')
        label, values = int(line[0]), line[1:]
        value = np.zeros((self.n_features))
        for item in values:
            idx, val = item.split(':')
            value[int(idx) - 1] = float(val)
        value[-1] = 1.0
        return label, value

    def __getitem__(self, index):
        with open(self.file_path, 'r') as fp:
            for idx, line in enumerate(fp):
                if idx == index:
                    return self.process_line(line.strip("\n"))

    def __len__(self):
        count = 0
        with open(self.file_path, 'r') as fp:
            for idx, line in enumerate(fp):
                count += 1
        return count

def build_train_test_set(dataloader, n_samples, n_features, ratio=0.8):
    n_train_samples = int(n_samples * ratio)
    n_test_samples = n_samples - n_train_samples
    perm = np.random.permutation(n_samples)
    train_sample_ids = perm[:n_train_samples]
    test_sample_ids = perm[n_train_samples:]
    train_features = np.zeros((n_train_samples, n_features))
    train_labels = np.zeros(n_train_samples)
    test_features = np.zeros((n_test_samples, n_features))
    test_labels = np.zeros(n_test_samples)
    train_id = 0
    test_id = 0
    pbar = tqdm(total=n_samples)
    for id, data in enumerate(dataloader):
        if id in train_sample_ids:
            label, feature = data
            label = label.numpy()[0]
            feature = feature.numpy()[0]
            label = 0 if label == 1 else 1  # map lables {1, 2} to {0, 1}
            train_features[train_id] = feature
            train_labels[train_id] = label
            train_id += 1
        elif id in test_sample_ids:
            label, feature = data
            label = label.numpy()[0]
            feature = feature.numpy()[0]
            label = 0 if label == 1 else 1  # map lables {1, 2} to {0, 1}
            test_features[test_id] = feature
            test_labels[test_id] = label
            test_id += 1
        pbar.update(1)
    pbar.close()
    return train_features, train_labels, test_features, test_labels

dataset_covtype_iter = LibsvmIterDataset('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale', 54)
dataloader_covtype_iter = DataLoader(dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
dataset_covtype = LibsvmDataset(
    '/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale',
    54)  # 用于按索引随机读取 covtype_binary 数据集

n_samples = len(dataset_covtype)
n_features = dataset_covtype_iter.n_features  # 55 (54 + 1)

file_path = '/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/full_data_label01.npz'

print('====== Prepare data =======')
if os.path.exists(file_path):
    print('====== Load sampled data from cache =======')
    data = np.load(file_path)
    train_features, train_labels, test_features, test_labels = data['train_features'], data['train_labels'], data['test_features'], data['test_labels']
else:
    train_features, train_labels, test_features, test_labels = build_train_test_set(dataloader_covtype_iter, n_samples, n_features, ratio=0.8)
    np.savez(file_path, train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
print('====== Done =======')

n_train_samples = train_features.shape[0]
n_test_samples = test_features.shape[0]

# 定义Loss函数
criterion = nn.BCELoss()

# 自定义随机梯度下降
def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data

# 使用Sequential定义神经网络
net = nn.Sequential(
    nn.Linear(n_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

def eval(model, test_features, test_labels):
    x = Variable(torch.from_numpy(test_features.astype(np.float32)))
    y = Variable(torch.from_numpy(test_labels.astype(np.float32)))
    y_pred = model(x)
    y_pred = torch.tensor(list(map(lambda e: 0 if e<=0.5 else 1, y_pred)))
    accuracy = torch.sum(y == y_pred) / len(y)
    return accuracy

# 开始训练
losses1 = []
idx = 0
start = time.time()
for e in range(5):
    train_loss = 0
    for i in range(1000):
        random_id = np.random.choice(n_train_samples)
        feature, label = train_features[random_id], train_labels[random_id]
        feature = Variable(torch.from_numpy(feature.astype(np.float32)))
        label = Variable(torch.Tensor([label]))
        # 前向传播
        out = net(feature)
        loss = criterion(out, label)
        # 反向传播
        net.zero_grad()
        loss.backward()
        sgd_update(net.parameters(), 1e-3)
        # 记录误差
        train_loss += loss.item()
        if idx % 30 == 0:
            losses1.append(loss.item())
        idx += 1
        acc = eval(net, test_features, test_labels)
        print('sample:{},Train Loss:{}, Test Acc: {}'.format(i, loss.item(), acc))
    print('epoch:{},Train Loss:{}'.format(e, train_loss / 100000))
end = time.time()
print('Time={}'.format(end - start))

x_axis = np.linspace(0, 5, len(losses1), endpoint=True)
plt.semilogy(x_axis, losses1, label='batch_size=1')
plt.show()