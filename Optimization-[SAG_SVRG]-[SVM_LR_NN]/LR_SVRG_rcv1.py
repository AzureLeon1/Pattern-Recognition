# -*-coding:utf-8 -*-
'''
@File    :   LR_SVRG_rcv1.py
@Time    :   2022/05/22 11:09:49
@Author  :   Liang Wang
@Contact :   wangliang.leon20@gmail.com
@Desc    :   逻辑回归 + SVRG优化算法 + rcv1_binary数据集
'''
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class LibsvmIterDataset(IterableDataset):
    def __init__(self, file_path, n_features):
        """LIBSVM格式数据顺序读取
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
        """LIBSVM格式数据随机读取
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

class LogisticRegression:
    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        self.n_features = n_features
        self._w = np.zeros(n_features)

    def fit_svrg(self, train_features, train_labels, test_features, test_labels, lr=0.01, epoch=100, inner_epoch=100, alpha=1.0):
        n_train_samples = train_features.shape[0]
        n_test_samples = test_features.shape[0]
        grad = 0  # 总梯度
        grad_sample = np.zeros((n_train_samples, self.n_features))  # 每个分量函数的梯度
        list_acc = []
        list_acc.append(0)
        list_train_obj_func = []
        list_test_obj_func = []
        # for epoch_id in tqdm(range(epoch)):
        for epoch_id in range(epoch):

            # 计算全梯度
            grad_all = alpha * self._w + np.mean((((self.predict(train_features, True) - train_labels).reshape(-1,1).repeat(n_features, axis=1)) * train_features), axis=0)
            w_saved = self._w.copy()

            # 内循环
            w_inner = np.zeros((inner_epoch, n_features))
            for inner_epoch_id in range(inner_epoch):
                random_id = np.random.choice(n_train_samples)
                y = train_labels[random_id]
                x = train_features[random_id]
                grad = alpha * self._w + (self.predict(x, True) - y) * x
                grad -= (alpha*w_saved + (self.predict_with_saved_w(x, w_saved)-y) * x - grad_all)
                self._w -= lr * grad / n_train_samples
                w_inner[inner_epoch_id] = self._w.copy()

                # ====== Train Eval [inner loop] =======
                accuracy = self.eval(test_features, test_labels)
                list_acc.append(accuracy)
                train_obj_func = 0.5 * alpha * np.linalg.norm(self._w) ** 2 - np.sum((train_labels * np.log(
                    self.predict(train_features, True)) + (1 - train_labels) * np.log(
                    1 - self.predict(train_features, True)))) / n_train_samples
                list_train_obj_func.append(train_obj_func)
                test_obj_func = 0.5 * alpha * np.linalg.norm(self._w) ** 2 - np.sum((test_labels * np.log(
                    self.predict(test_features, True)) + (1 - test_labels) * np.log(
                    1 - self.predict(test_features, True)))) / n_test_samples
                list_test_obj_func.append(test_obj_func)
                # print('Epoch: {}, Test Acc: {}, v1: {}, v2: {}'.format(epoch_id, accuracy, train_obj_func, test_obj_func))

            # 取内循环迭代的参数的均值进行更新
            self._w = np.mean(w_inner, axis=0)
            # ====== Train Eval [outer loop] =======
            accuracy = self.eval(test_features, test_labels)
            list_acc.append(accuracy)
            train_obj_func = 0.5 * alpha * np.linalg.norm(self._w)**2 - np.sum((train_labels * np.log(self.predict(train_features, True)) + (1 - train_labels) * np.log(1 - self.predict(train_features, True)))) / n_train_samples
            list_train_obj_func.append(train_obj_func)
            test_obj_func = 0.5 * alpha * np.linalg.norm(self._w)**2 - np.sum((test_labels * np.log(self.predict(test_features, True)) + (1 - test_labels) * np.log(1 - self.predict(test_features, True)))) / n_test_samples
            list_test_obj_func.append(test_obj_func)
            # print('Epoch: {}, Test Acc: {}, v1: {}, v2: {}'.format(epoch_id, accuracy, train_obj_func, test_obj_func))

        return list_acc, list_train_obj_func, list_test_obj_func


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        y_pred = self.sigmoid(x @ self._w)
        if raw:
            return y_pred
        return list(map(lambda x: 0 if x<=0.5 else 1, y_pred))

    def predict_with_saved_w(self, x, w_saved):
        x = np.asarray(x, np.float32)
        y_pred = self.sigmoid(x @ w_saved)
        return y_pred

    def eval(self, test_features, test_labels):
        x = test_features
        y = test_labels
        y_pred = self.predict(x, raw=False)
        accuracy = (y==y_pred).sum() / len(y)
        return accuracy


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
        # print(id)
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



if __name__=='__main__':
    # 读取数据
    file_path = '/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/sampled_data_label01.npz'

    print('====== Load sampled data from cache =======')
    data = np.load(file_path)
    train_features, train_labels, test_features, test_labels = data['train_features'], data['train_labels'], data[
        'test_features'], data['test_labels']
    print('====== Done =======')

    n_train_samples = train_features.shape[0]
    n_test_samples = test_features.shape[0]
    n_samples = n_train_samples + n_test_samples
    n_features = train_features.shape[1]

    # 模型训练与评估
    clf = LogisticRegression(n_samples=n_samples, n_features=n_features)
    print('====== Train =======')
    train_acc, list_train_obj_func, list_test_obj_func = clf.fit_svrg(train_features, train_labels, test_features, test_labels, lr=0.1, epoch=100, inner_epoch=100, alpha=0)
    print('====== Eval =======')
    print('Best Accuracy: {:.2%}'.format(max(train_acc)))

    # 绘制实验结果
    import datetime
    import time

    stamp = int(time.time())
    s_time = str(datetime.datetime.fromtimestamp(stamp))[11:]

    plt.xlabel("#Epoch")
    plt.title("[LR_SVRG] Accuracy")
    plt.plot(range(len(train_acc)), train_acc)
    plt.show()
    fig_name = './img/lr_svrg_acc_{}.jpg'.format(s_time)
    plt.savefig(fig_name)

    plt.xlabel("#Epoch")
    plt.title("[LR_SVRG] Objective on training set")
    plt.plot(range(len(list_train_obj_func)), list_train_obj_func)
    plt.show()
    fig_name = './img/lr_svrg_obj_train_{}.jpg'.format(s_time)
    plt.savefig(fig_name)

    plt.xlabel("#Epoch")
    plt.title("[LR_SVRG] Objective on testing set")
    plt.plot(range(len(list_test_obj_func)), list_test_obj_func)
    plt.show()
    fig_name = './img/lr_svrg_obj_test_{}.jpg'.format(s_time)
    plt.savefig(fig_name)

