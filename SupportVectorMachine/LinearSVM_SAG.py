import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np

class LibsvmIterDataset(IterableDataset):
    def __init__(self, file_path, n_features):
        """
        file_path: Libsvm格式数据文件地址
        n_features: 特征数，从1开始
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

    def fit_sg(self, dataloader, c=1, lr=0.01, epoch=10000, lambda_=2.0):
        #         x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        n_train_samples = int(self.n_samples * self.split_ratio)
        grad = 0  # 总梯度
        grad_sample = np.zeros((n_train_samples, self.n_features))  # 每个分量函数的梯度
        dataiter = iter(dataloader)
        next(dataiter)
        idx = 0
        for id in range(epoch):
            print('Train {}'.format(id))
            # 注意即使所有 x, y 都满足 w·x + b >= 1
            # 由于损失里面有一个 w 的模长平方
            # 所以仍然不能终止训练，只能截断当前的梯度下降
            self._w *= 1 - lr

            step = np.random.choice(50) + 1
            for i in range(step):
                try:
                    idx += 1
                    if idx >= n_train_samples:
                        raise StopIteration
                    y, x = next(dataiter)
                except StopIteration:
                    # restart
                    idx = 0
                    dataiter = iter(dataloader)
                    y, x = next(dataiter)
            y = y.numpy()[0]
            x = x.numpy()[0]
            err = 1 - y * self.predict(x, True)
            if err <= 0:
                continue

            new_grad_sample = - c * y * x
            grad = grad - grad_sample[idx] + new_grad_sample
            grad_sample[idx] = new_grad_sample

            self._w -= lr * new_grad_sample / n_train_samples
            # self._b += delta

    def fit_sag(self, dataloader, c=1, lr=0.01, epoch=10000, lambda_=2.0):
        #         x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        n_train_samples = int(self.n_samples * self.split_ratio)
        grad = 0  # 总梯度
        grad_sample = np.zeros((n_train_samples, self.n_features))  # 每个分量函数的梯度
        dataiter = iter(dataloader)
        next(dataiter)
        idx = 0
        for id in range(epoch):
            print('Train {}'.format(id))
            # 注意即使所有 x, y 都满足 w·x + b >= 1
            # 由于损失里面有一个 w 的模长平方
            # 所以仍然不能终止训练，只能截断当前的梯度下降

            step = np.random.choice(50) + 1
            for i in range(step):
                try:
                    idx += 1
                    if idx >= n_train_samples:
                        raise StopIteration
                    y, x = next(dataiter)
                except StopIteration:
                    # restart
                    idx = 0
                    dataiter = iter(dataloader)
                    y, x = next(dataiter)
            y = y.numpy()[0]
            x = x.numpy()[0]
            err = 1 - y * self.predict(x, True)
            if err <= 0:
                new_grad_sample = self._w
            else:
                new_grad_sample = self._w - c * y * x
            grad = grad - grad_sample[idx] + new_grad_sample
            grad_sample[idx] = new_grad_sample

            self._w -= lr * grad / n_train_samples
            # self._b += delta

    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
        # y_pred = x.dot(self._w) + self._b
        y_pred = x.dot(self._w)
        if raw:
            return y_pred
        # return np.sign(y_pred).astype(np.float32)  # because ground truth is in [1, 2]
        return  1 if y_pred <=1.5 else 2  # because ground truth is in [1, 2]

    def eval(self, dataloader, n_test_sample=1000):
        cnt = 0
        cnt_right = 0
        n_train_samples = int(self.n_samples * self.split_ratio)

        dataiter = iter(dataloader)
        idx = 0
        for i in range(n_train_samples):
            next(dataiter)
            idx += 1
        for id in range(n_test_sample):
            print('Test {}'.format(id))
            step = np.random.choice(10) + 1
            for i in range(step):
                try:
                    idx += 1
                    y, x = next(dataiter)
                except StopIteration:
                    # restart
                    dataiter = iter(dataloader)
                    idx = 0
                    for i in range(n_train_samples):
                        next(dataiter)
                        idx += 1
                    y, x = next(dataiter)
            y_pred = self.predict(x, raw=False)
            cnt += 1
            if y_pred == y:
                cnt_right += 1

        # for idx, data in enumerate(dataloader):
        #     if idx < n_train_samples:  # skip training samples
        #         continue
        #     if idx >= n_train_samples + n_test_sample:
        #         break
        #     print('test {}'.format(idx))
        #     y, x = data
        #     y_pred = self.predict(x, raw=False)
        #     cnt += 1
        #     if y_pred == y:
        #         cnt_right += 1
        return cnt_right / cnt

if __name__=='__main__':
    dataset_covtype_iter = LibsvmIterDataset('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale', 54)
    dataloader_covtype_iter = DataLoader(dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
    dataset_covtype = LibsvmDataset('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale', 54)  # 用于按索引随机读取 covtype_binary 数据集

    n_samples = len(dataset_covtype)
    n_features = dataset_covtype_iter.n_features  # 55 (54 + 1)


    svm = LinearSVM(n_samples=n_samples, n_features=n_features, split_ratio=0.7)
    svm.fit_sg(dataloader_covtype_iter, epoch=100000)
    accuracy = svm.eval(dataset_covtype_iter, n_test_sample=2000)
    print('Test Accuracy: {:.2%}'.format(accuracy))