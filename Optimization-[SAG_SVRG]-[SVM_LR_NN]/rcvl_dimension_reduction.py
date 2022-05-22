'''
@File    :   rcvl_dimension_reduction.py
@Time    :   2022/05/22 11:04:04
@Author  :   Liang Wang
@Contact :   wangliang.leon20@gmail.com
@Desc    :   对 rcv1_binary 数据集进行数据采样和特征降维
'''
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset, DataLoader

def build_dataset(dataloader, n_samples, n_features):
    """根据 Dataloader 构造 numpy.array 格式的数据
    """
    features = np.zeros((n_samples, n_features))
    labels = np.zeros(n_samples)
    pbar = tqdm(total=n_samples)
    for id, data in enumerate(dataloader):
        label, feature = data
        label = label.numpy()[0]
        label = 0 if label == -1 else 1  # map lables {-1, 1} to {0, 1}
        feature = feature.numpy()[0]
        features[id] = feature
        labels[id] = label
        pbar.update(1)
    pbar.close()
    return features, labels

class LibsvmIterDataset(IterableDataset):
    def __init__(self, file_path, n_features):
        """LIBSVM格式数据顺序读取
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
        """LIBSVM格式数据随机读取
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

train_dataset_covtype_iter = LibsvmIterDataset('/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/rcv1_train.binary', 47236)
train_dataloader_covtype_iter = DataLoader(train_dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
train_dataset_covtype = LibsvmDataset('/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/rcv1_train.binary', 47236)  # 用于按索引随机读取 covtype_binary 数据集

test_dataset_covtype_iter = LibsvmIterDataset('/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/rcv1_test.binary', 47236)
test_dataloader_covtype_iter = DataLoader(test_dataset_covtype_iter, batch_size=1)   # 用于顺序迭代 covtype_binary 数据集
test_dataset_covtype = LibsvmDataset('/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/rcv1_test.binary', 47236)  # 用于按索引随机读取 covtype_binary 数据集

n_train_samples = len(train_dataset_covtype)
n_test_samples = len(test_dataset_covtype)
n_samples = n_train_samples + n_test_samples
n_features = train_dataset_covtype_iter.n_features  # 55 (54 + 1)

# 从 libsvm 格式文件读取数据，生成 numpy.array
train_features, train_labels = build_dataset(train_dataloader_covtype_iter, n_train_samples, n_features)
test_features, test_labels = build_dataset(test_dataloader_covtype_iter, n_test_samples, n_features)

# 数据采样：8000个训练样本，2000个测试样本
n_selected_train_samples = 8000
id = np.random.permutation(n_train_samples)[:8000]
selected_train_features, selected_train_labels = train_features[id], train_labels[id]
n_selected_test_samples = 2000
id = np.random.permutation(n_test_samples)[:2000]
selected_test_features, selected_test_labels = test_features[id], test_labels[id]

print(selected_train_features.shape)
print(selected_train_labels.shape)
print(selected_test_features.shape)
print(selected_test_labels.shape)

# 采用 PCA 进行特征降维
from sklearn.decomposition import PCA
pca = PCA(n_components=20)

train_features_pca = pca.fit_transform(train_features)

selected_train_features_pca = pca.transform(selected_train_features)
selected_test_features_pca = pca.transform(selected_test_features)

print(selected_train_features_pca.shape)
print(selected_test_features_pca.shape)

# 保存采样和降维后的数据
file_path = '/home/wangliang/datasets/Pattern-Recogniition/rcv1_binary/sampled_data_label01.npz'
np.savez(file_path, train_features=selected_train_features_pca, train_labels=selected_train_labels, test_features=selected_test_features_pca, test_labels=selected_test_labels)

