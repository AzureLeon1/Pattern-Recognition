import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class LibsvmIterDataset(IterableDataset):
    def __init__(self, file_path, n_features):
        """
        file_path: Libsvm格式数据文件地址
        n_features: 特征数
        """
        self.file_path = file_path
        self.n_features = n_features + 1  # augment feature

    def process_line(self, line):
        # print(line)
        line = line.strip().split(' ')
#         print(line)
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
        #         print(line)
        line = line.strip().split(' ')
        #         print(line)
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


class LinearSVM:
    def __init__(self, n_samples, n_features, split_ratio=0.7):
        self.n_samples = n_samples
        self.n_features = n_features
        self.split_ratio = split_ratio
        self._w = np.zeros(n_features)
        # self._b = 0.

    def fit_sag(self, train_features, train_labels, test_features, test_labels, c=1, lr=0.01, epoch=10000, lambda_=1.0):
        #         x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        n_train_samples = train_features.shape[0]
        grad = 0  # 总梯度
        grad_sample = np.zeros((n_train_samples, self.n_features))  # 每个分量函数的梯度
        idx = 0
        list_acc = []
        for _ in tqdm(range(epoch)):
            # 注意即使所有 x, y 都满足 w·x + b >= 1
            # 由于损失里面有一个 w 的模长平方
            # 所以仍然不能终止训练，只能截断当前的梯度下降
            self._w *= 1 - lr

            random_id = np.random.choice(n_train_samples)
            y = train_labels[random_id]
            x = train_features[random_id]
            err = 1 - y * self.predict(x, True)
            if err <= 0:
                continue

            new_grad_sample = - c * y * x
            grad = grad - grad_sample[idx] + new_grad_sample
            grad_sample[idx] = new_grad_sample

            self._w -= lr * grad
            # self._w -= lr * grad + 2 * lambda_ * lr * self._w
            # self._b += delta
            # ====== Train Eval =======
            accuracy = self.eval(test_features, test_labels)
            list_acc.append(accuracy)
        return list_acc


    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        # y_pred = x.dot(self._w) + self._b
        y_pred = x.dot(self._w)
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)  # because ground truth is in [1, 2]

    def eval(self, test_features, test_labels):
        x = test_features
        y = test_labels
        y_pred = self.predict(x, raw=False)
        accuracy = (y==y_pred).sum() / len(y)
        return accuracy

def build_train_test_set(dataloader, n_samples, n_features, n_need_samples=10000, ratio=0.8):
    n_train_samples = int(n_samples * ratio)
    n_need_train_samples = int(n_need_samples * ratio)
    train_sample_ids = np.random.choice(range(n_train_samples), n_need_train_samples, replace=False)
    test_sample_ids = np.random.choice(range(n_train_samples, n_samples), n_need_samples - n_need_train_samples, replace=False)
    # print(train_sample_ids)
    # print(test_sample_ids)
    train_features = np.zeros((n_need_train_samples, n_features))
    train_labels = np.zeros(n_need_train_samples)
    test_features = np.zeros((n_need_samples - n_need_train_samples, n_features))
    test_labels = np.zeros(n_need_samples - n_need_train_samples)
    train_id = 0
    test_id = 0
    pbar = tqdm(total=n_samples)
    for id, data in enumerate(dataloader):
        # print(id)
        if id in train_sample_ids:
            label, feature = data
            label = label.numpy()[0]
            feature = feature.numpy()[0]
            label = -1. if label == 1 else 1.  # map lables {1, 2} to {-1, 1}
            train_features[train_id] = feature
            train_labels[train_id] = label
            train_id += 1
        elif id in test_sample_ids:
            label, feature = data
            label = label.numpy()[0]
            feature = feature.numpy()[0]
            label = -1. if label == 1 else 1.  # map lables {1, 2} to {-1, 1}
            test_features[test_id] = feature
            test_labels[test_id] = label
            test_id += 1
        pbar.update(1)
    pbar.close()
    return train_features, train_labels, test_features, test_labels



if __name__=='__main__':
    dataset_covtype_iter = LibsvmIterDataset('../datasets/covtype_binary/covtype.libsvm.binary.scale', 54)
    dataloader_covtype_iter = DataLoader(dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
    dataset_covtype = LibsvmDataset('../datasets/covtype_binary/covtype.libsvm.binary.scale', 54)  # 用于按索引随机读取 covtype_binary 数据集

    n_samples = len(dataset_covtype)
    n_features = dataset_covtype_iter.n_features  # 55 (54 + 1)

    n_need_samples = 100000

    file_path = '../datasets/covtype_binary/sampled_data_{}.npz'.format(n_need_samples)

    print('====== Sample data =======')
    if os.path.exists(file_path):
        print('====== Load sampled data from cache =======')
        data = np.load(file_path)
        train_features, train_labels, test_features, test_labels = data['train_features'], data['train_labels'], data['test_features'], data['test_labels']
    else:
        train_features, train_labels, test_features, test_labels = build_train_test_set(dataloader_covtype_iter, n_samples, n_features, n_need_samples=n_need_samples, ratio=0.8)
        np.savez(file_path, train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)

    # svm = LinearSVM(n_samples=n_samples, n_features=n_features)
    # print('====== Train =======')
    # train_acc = svm.fit_sag(train_features, train_labels, test_features, test_labels, c=1, lr=0.001, epoch=20000, lambda_=1.0/n_features)
    # print('====== Eval =======')
    # accuracy = svm.eval(test_features, test_labels)
    # print('Test Accuracy: {:.2%}'.format(accuracy))
    #
    # # print(train_acc)
    # plt.xlabel("#Epoch")
    # plt.xlabel("#Accuracy")
    # plt.plot(range(len(train_acc)), train_acc)
    # plt.show()

    # TODO: 验证逻辑回归 sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, tol=1e-3))
    clf.fit(train_features, train_labels)
    y_pred = clf.predict(test_features)
    accuracy = (test_labels == y_pred).sum() / len(test_labels)
    print(accuracy)

