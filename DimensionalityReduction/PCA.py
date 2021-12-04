import numpy as np
from numpy.linalg import eig
from scipy.io import loadmat
import matplotlib.pyplot as plt

class PCA():
    def calculate_covariance_matrix(self, X, Y=None):
        """
        计算协方差矩阵
        """
        m = X.shape[0]
        X = X - np.mean(X, axis=0)
        Y = X if Y == None else Y - np.mean(Y, axis=0)
        return 1 / m * np.matmul(X.T, Y)

    def transform(self, X, n_components):
        """
        对全部数据进行PCA变换
        设n=X.shape[1]，将n维数据降维成n_component维
        """
        covariance_matrix = self.calculate_covariance_matrix(X)

        # 获取特征值，和特征向量
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # 对特征向量排序，并取最大的前n_component组
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]

        # 转换
        return np.real(np.matmul(X, eigenvectors))

    def transform_train_test(self, X_train, X_test, n_components):
        """
        只用训练集计算特征向量，用算出的特征向量分别对训练集和测试进行变换
        """
        covariance_matrix = self.calculate_covariance_matrix(X_train)

        # 获取特征值，和特征向量
        eigenvalues, eigenvectors = eig(covariance_matrix)

        # 对特征向量排序，并取最大的前n_component组
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvectors = eigenvectors[:, :n_components]

        # 转换
        return np.real(np.matmul(X_train, eigenvectors)), np.real(np.matmul(X_test, eigenvectors))


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

        new_train_features_ORL, new_test_features_ORL = PCA().transform_train_test(train_features_ORL,
                                                                                   test_features_ORL, dim)

        predictions_ORL = KNNClassifier(n_neighbors=1).predict(new_train_features_ORL, train_labels_ORL,
                                                               new_test_features_ORL)
        acc = accuracy(test_labels_ORL, predictions_ORL)
        accs.append(acc)
        print('Dataset: ORL, Dims:', dim, ', Acc:', acc)

    dims.append(train_features_ORL.shape[1])
    predictions_ORL = KNNClassifier(n_neighbors=1).predict(train_features_ORL, train_labels_ORL, test_features_ORL)
    acc = accuracy(test_labels_ORL, predictions_ORL)
    accs.append(acc)

    plt.plot(dims, accs, 'r')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy on ORL Dataset')
    plt.title('PCA-KNN-ORL')
    plt.savefig("./img/PCA-KNN-ORL.png", dpi=300)
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

        new_train_features_vehicle, new_test_features_vehicle = PCA().transform_train_test(train_features_vehicle,
                                                                                   test_features_vehicle, dim)

        predictions_vehicle = KNNClassifier(n_neighbors=1).predict(new_train_features_vehicle, train_labels_vehicle,
                                                               new_test_features_vehicle)
        acc = accuracy(test_labels_vehicle, predictions_vehicle)
        accs.append(acc)
        print('Dataset: vehicle, Dims:', dim, ', Acc:', acc)

    dims.append(train_features_vehicle.shape[1])
    predictions_vehicle = KNNClassifier(n_neighbors=1).predict(train_features_vehicle, train_labels_vehicle, test_features_vehicle)
    acc = accuracy(test_labels_vehicle, predictions_vehicle)
    accs.append(acc)

    plt.plot(dims, accs, 'r')
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy on vehicle Dataset')
    plt.title('PCA-KNN-vehicle')
    plt.savefig("./img/PCA-KNN-vehicle.png", dpi=300)
    plt.show()