import numpy as np
from numpy.linalg import eig
from scipy.io import loadmat
import matplotlib.pyplot as plt

class LDA():
    def __init__(self):
        self.Xi_means = 0        # 每个类别的均值向量
        self.X_means = 0         # 整体均值向量
        self.Xi_cov_matrix = []  # 每个类别的协方差矩阵
        self.X_cov_matrix = 0    # 整体的协方差矩阵
        self.features = 0
        self.labels = 0
        self.classes = 0        # 所有类别
        self.priors = 0         # 每个类别的先验概率
        self.n_samples = 0      # 样本数目
        self.n_features = 0     # 特征维数
        self.n_components = 0
        self.w = 0              # 投影向量
        self.bincount_ = 0      # 每个类别的数目

    def params_init(self, features, labels):
        self.features = features
        self.labels = labels

        self.n_samples, self.n_features = features.shape

        self.classes, yidx = np.unique(labels, return_inverse=True)

        self.bincount_ = np.bincount(labels)
        if len(self.bincount_) == len(self.classes) + 1:
            self.bincount_ = self.bincount_[1:]
            assert len(self.bincount_) == len(self.classes)
        self.priors = self.bincount_ / self.n_samples

        means = np.zeros((len(self.classes), self.n_features))
        np.add.at(means, yidx, features)
        self.Xi_means = means / np.expand_dims(self.bincount_, 1)

        self.Xi_cov_matrix = [np.cov(features[labels == label_value].T) for label_idx, label_value in enumerate(self.classes)]

        self.X_cov_matrix = sum(self.Xi_cov_matrix) / len(self.Xi_cov_matrix)

        self.X_means = np.dot(np.expand_dims(self.priors, axis=0), self.Xi_means)
        return

    def transform_train_test(self, train_features, train_labels, test_features, n_components):
        """只用训练集计算特征向量，用算出的特征向量分别对测试集进行变换"""
        self.params_init(train_features, train_labels)
        Sw = self.X_cov_matrix
        Sb = sum([self.bincount_[class_idx]*np.dot((self.Xi_means[class_idx,None] - self.X_means).T, (self.Xi_means[class_idx,None] - self.X_means)) \
                for class_idx, class_value in enumerate(self.classes)]) / (self.n_samples - 1)
        #SVD求Sw的逆矩阵
        Swn = np.linalg.inv(Sw + 1e-5 * np.eye(Sw.shape[0]))
        # U,S,V = np.linalg.svd(Sw)
        # Sn = np.linalg.inv(np.diag(S))
        # Swn = np.dot(np.dot(V.T,Sn),U.T)
        SwnSb = np.dot(Swn,Sb)
        #求特征值和特征向量，并取实数部分
        la,vectors = np.linalg.eig(SwnSb)
        la = np.real(la)
        vectors = np.real(vectors)
        #特征值的下标从大到小排列
        laIdx = np.argsort(-la)
        #默认选取(N-1)个特征值的下标
        if n_components == None:
            n_components = len(self.classes_)-1
        #选取特征值和向量
        lambda_index = laIdx[:n_components]
        w = vectors[:,lambda_index]
        self.w = w
        self.n_components = n_components
        return train_features @ self.w, test_features @ self.w

class KNNClassifier():
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def cal_all_distances(self, train_features, test_feature):
        """
        计算一个测试样本到所有训练样本之间的欧氏距离
        :param train_features: 训练样本集合
        :param test_feature: 测试样本
        :return: 距离
        """
        distances = []
        for train_feature in train_features:
            dist = np.linalg.norm(test_feature - train_feature)
            distances.append(dist)
        return distances

    def predict(self, train_features, train_labels, test_features):
        """
        对训练样本进行预测
        :param train_features: 训练样本特征
        :param train_labels: 训练样本标签
        :param test_features: 测试样本特征
        :return: 测试样本预测标签
        """
        n_test_samples = test_features.shape[0]
        neighbors = []
        for i in range(n_test_samples):
            distances = self.cal_all_distances(train_features, test_features[i])
            k_nearest_neighbors = np.array(distances).argsort()[:self.n_neighbors]
            neighbors.append(k_nearest_neighbors)
        neighbors = np.array(neighbors)
        if self.n_neighbors == 1: # 最近邻算法
            neighbors_idx = neighbors.reshape(-1)
            return train_labels[[neighbors_idx]]
        else:
            neighbors_idx = []
            for neighbor in neighbors:
                idx = np.argmax(np.bincount(neighbor))
                neighbors_idx.append(idx)
            return train_labels[[neighbors_idx]]


def load_ORL():
    data_path = '../datasets/ORL-AT&T/ORLData_25.mat'
    data = loadmat(data_path)
    data = data['ORLData']
    data = data.astype(np.int32) # uint8 -> int32
    data = data.T
    np.random.shuffle(data)
    return data


def load_vehicle():
    data_path = '../datasets/Vehicle-UCI/vehicle.mat'
    data = loadmat(data_path)
    data = data['UCI_entropy_data']['train_data'][0,0]
    data = data.astype(np.int32) # uint8 -> int32
    data = data.T
    np.random.shuffle(data)
    return data


def accuracy(ground_truth, prediction):
    return np.mean(np.equal(ground_truth, prediction))

if __name__ == "__main__":
    data_ORL = load_ORL()
    data_vehicle = load_vehicle()

    # ORL 数据集
    dims = list(range(5, 601, 5))
    accs = []
    for dim in dims:
        features_ORL = data_ORL[:, :-1]
        labels_ORL = data_ORL[:, -1]

        n_samples = data_ORL.shape[0]
        partition = int(0.8 * n_samples)

        train_features_ORL, test_features_ORL = features_ORL[:partition], features_ORL[partition:]
        train_labels_ORL, test_labels_ORL = labels_ORL[:partition], labels_ORL[partition:]

        new_train_features_ORL, new_test_features_ORL = LDA().transform_train_test(train_features_ORL, train_labels_ORL,
                                                                                   test_features_ORL, dim)

        predictions_ORL = KNNClassifier(n_neighbors=1).predict(new_train_features_ORL, train_labels_ORL,
                                                               new_test_features_ORL)
        acc = accuracy(test_labels_ORL, predictions_ORL)
        accs.append(acc)
        print('Dataset: ORL, Dims:', dim, ', Acc:', acc)

    # dims.append(train_features_ORL.shape[1])
    # predictions_ORL = KNNClassifier(n_neighbors=1).predict(train_features_ORL, train_labels_ORL, test_features_ORL)
    # acc = accuracy(test_labels_ORL, predictions_ORL)
    # accs.append(acc)

    plt.plot(dims, accs, 'r')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy on ORL Dataset')
    plt.title('LDA-KNN-ORL')
    plt.savefig("./img/LDA-KNN-ORL.png", dpi=300)
    plt.show()

    # vehicle 数据集
    dims = list(range(1, 18))
    accs = []
    for dim in dims:
        features_vehicle = data_vehicle[:, :-1]
        labels_vehicle = data_vehicle[:, -1]

        n_samples = data_vehicle.shape[0]
        partition = int(0.8 * n_samples)

        train_features_vehicle, test_features_vehicle = features_vehicle[:partition], features_vehicle[partition:]
        train_labels_vehicle, test_labels_vehicle = labels_vehicle[:partition], labels_vehicle[partition:]

        new_train_features_vehicle, new_test_features_vehicle = LDA().transform_train_test(train_features_vehicle, train_labels_vehicle,
                                                                                   test_features_vehicle, dim)

        predictions_vehicle = KNNClassifier(n_neighbors=1).predict(new_train_features_vehicle, train_labels_vehicle,
                                                               new_test_features_vehicle)
        acc = accuracy(test_labels_vehicle, predictions_vehicle)
        accs.append(acc)
        print('Dataset: vehicle, Dims:', dim, ', Acc:', acc)

    # dims.append(train_features_vehicle.shape[1])
    # predictions_vehicle = KNNClassifier(n_neighbors=1).predict(train_features_vehicle, train_labels_vehicle, test_features_vehicle)
    # acc = accuracy(test_labels_vehicle, predictions_vehicle)
    # accs.append(acc)

    plt.plot(dims, accs, 'r')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy on vehicle Dataset')
    plt.title('LDA-KNN-vehicle')
    plt.savefig("./img/LDA-KNN-vehicle.png", dpi=300)
    plt.show()