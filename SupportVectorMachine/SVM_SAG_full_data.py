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

    def fit_sag(self, train_features, train_labels, test_features, test_labels, c=1, lr=0.01, epoch=10000, alpha=0.01):
        #         x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        n_train_samples = train_features.shape[0]
        n_test_samples = test_features.shape[0]
        grad = 0  # 总梯度
        grad_sample = np.zeros((n_train_samples, self.n_features))  # 每个分量函数的梯度
        list_acc = []
        list_train_obj_func = []
        list_test_obj_func = []
        for _ in tqdm(range(epoch)):
            # self._w -= lr * alpha * self._w
            #
            # random_id = np.random.choice(n_train_samples)
            # y = train_labels[random_id]
            # x = train_features[random_id]
            # err = 1 - y * self.predict(x, True)
            # if err <= 0:
            #     continue
            # new_grad_sample = - c * y * x
            # grad = grad - grad_sample[random_id] + new_grad_sample
            # grad_sample[random_id] = new_grad_sample
            #
            # self._w -= lr * grad


            random_id = np.random.choice(n_train_samples)
            y = train_labels[random_id]
            x = train_features[random_id]
            err = 1 - y * self.predict(x, True)
            if err <= 0:
                new_grad_sample = alpha * self._w
            else:
                new_grad_sample = alpha * self._w - c * y * x
            grad = grad - grad_sample[random_id] + new_grad_sample
            grad_sample[random_id] = new_grad_sample
            self._w -= lr / n_train_samples * grad

            # self._w -= lr * grad + 2 * lambda_ * lr * self._w
            # ====== Train Eval =======
            accuracy = svm.eval(test_features, test_labels)
            print('acc: {}'.format(accuracy))
            list_acc.append(accuracy)
            term1 = 0.5 * alpha  * np.linalg.norm(self._w)**2
            term2 = c * np.mean(np.maximum(1-train_labels * (train_features @ self._w), np.zeros(n_train_samples)))
            train_obj_func = term1 + term2
            print('value: {}, term 1: {}, term 2: {}'.format(train_obj_func, term1, term2))
            list_train_obj_func.append(train_obj_func)
            test_obj_func = 0.5 * alpha * np.linalg.norm(self._w)**2 + c * np.mean(np.maximum(1-test_labels * (test_features @ self._w), np.zeros(n_test_samples)))
            list_test_obj_func.append(test_obj_func)
        return list_acc, list_train_obj_func, list_test_obj_func


    def predict(self, x, raw=False):
        x = np.asarray(x, np.float32)
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

def build_train_test_set(dataloader, n_samples, n_features, ratio=0.8):
    n_train_samples = int(n_samples * ratio)
    n_test_samples = n_samples - n_train_samples
    perm = np.random.permutation(n_samples)
    train_sample_ids = perm[:n_train_samples]
    test_sample_ids = perm[n_train_samples:]
    # print(train_sample_ids)
    # print(test_sample_ids)
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
    dataset_covtype_iter = LibsvmIterDataset('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale', 54)
    dataloader_covtype_iter = DataLoader(dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
    dataset_covtype = LibsvmDataset('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale', 54)  # 用于按索引随机读取 covtype_binary 数据集

    n_samples = len(dataset_covtype)
    n_features = dataset_covtype_iter.n_features  # 55 (54 + 1)

    file_path = '/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/full_data.npz'

    print('====== Prepare data =======')
    if os.path.exists(file_path):
        print('====== Load sampled data from cache =======')
        data = np.load(file_path)
        train_features, train_labels, test_features, test_labels = data['train_features'], data['train_labels'], data['test_features'], data['test_labels']
    else:
        train_features, train_labels, test_features, test_labels = build_train_test_set(dataloader_covtype_iter, n_samples, n_features, ratio=0.8)
        np.savez(file_path, train_features=train_features, train_labels=train_labels, test_features=test_features, test_labels=test_labels)
    print('====== Done =======')

    svm = LinearSVM(n_samples=n_samples, n_features=n_features)
    print('====== Train =======')
    train_acc, list_train_obj_func, list_test_obj_func = svm.fit_sag(train_features, train_labels, test_features, test_labels, c=1, lr=0.1, epoch=100000, alpha=1)
    print('====== Eval =======')
    # accuracy = svm.eval(test_features, test_labels)
    # print('Test Accuracy: {:.2%}'.format(accuracy))
    print('Best Accuracy: {:.2%}'.format(max(train_acc)))


    # print(train_acc)
    plt.xlabel("#Epoch")
    plt.ylabel("#Accuracy")
    plt.plot(range(len(train_acc)), train_acc)
    plt.show()

    plt.xlabel("#Epoch")
    plt.title("Objective function value on training set")
    plt.plot(range(len(list_train_obj_func)), list_train_obj_func)
    plt.show()

    plt.xlabel("#Epoch")
    plt.title("Objective function value on testing set")
    plt.plot(range(len(list_test_obj_func)), list_test_obj_func)
    plt.show()

    # 验证sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    # 0.762
    # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=300000, tol=1e-3))
    # clf.fit(train_features, train_labels)
    # y_pred = clf.predict(test_features)
    # accuracy = (test_labels == y_pred).sum() / len(test_labels)
    # print(accuracy)

    # 0.761
    # clf = SGDClassifier(max_iter=6000, tol=1e-3)
    # clf.fit(train_features, train_labels)
    # y_pred = clf.predict(test_features)
    # accuracy = (test_labels == y_pred).sum() / len(test_labels)
    # print(accuracy)

    # 0.756
    # clf = LogisticRegression(penalty='l2')
    # clf.fit(train_features, train_labels)
    # y_pred = clf.predict(test_features)
    # accuracy = (test_labels == y_pred).sum() / len(test_labels)
    # print(accuracy)

    # clf = SGDClassifier(loss='log', max_iter=6000, tol=1e-3)
    # clf.fit(train_features, train_labels)
    # y_pred = clf.predict(test_features)
    # accuracy = (test_labels == y_pred).sum() / len(test_labels)
    # print(accuracy)

    # 0.75
    # clf = SGDClassifier(loss='log', max_iter=500, tol=1e-3)
    # clf.fit(train_features, train_labels)
    # accuracy = clf.score(test_features, test_labels)
    # print(accuracy)




