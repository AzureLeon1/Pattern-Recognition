import struct
import os
import math
import numpy as np


def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
    images=np.empty((imgNum , 784))#创建空数组
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images


def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def calculate_covariance_matrix(x):
    # 计算协方差矩阵
    n = x.shape[0]
    m = x.shape[1]
    x = x - np.repeat(np.mean(x, axis=1), m, axis=0).reshape(n, -1)
    return (1 / m) * np.matmul(x, x.T)


# def calculate_weight():


def loadDataset():
    data_dir = os.path.join('../../../datasets', 'MNIST')
    train_x_filename=os.path.join(data_dir, "train-images-idx3-ubyte")
    train_y_filename=os.path.join(data_dir, "train-labels-idx1-ubyte")
    test_x_filename=os.path.join(data_dir, "t10k-images-idx3-ubyte")
    test_y_filename=os.path.join(data_dir, "t10k-labels-idx1-ubyte")
    train_x=read_image(train_x_filename)#60000*784 的矩阵
    train_y=read_label(train_y_filename)#60000*1的矩阵
    test_x=read_image(test_x_filename)#10000*784
    test_y=read_label(test_y_filename)#10000*1

    return train_x, test_x, train_y, test_y


def calculate_weights(train_x, class_num):
    x_dim = train_x.shape[1]
    class_rate = []
    # class_x = []
    cov_mat = []
    mean_vec = []
    w1 = []  # 一次项系数向量
    w2 = []  # 常数项
    for i in range(class_num):
        class_is_i_index = np.where(train_y == i)[0]

        i_rate = len(class_is_i_index) / len(train_y)
        class_rate.append(i_rate)

        class_is_i_x = np.array([train_x[x] for x in class_is_i_index]).T
        # class_x.append(class_is_i_x)

        cov_mat_i = calculate_covariance_matrix(class_is_i_x)
        cov_mat.append(cov_mat_i)

        mean_vec_i = np.mean(class_is_i_x, axis=1)
        mean_vec.append(mean_vec_i)

    # 先验概率归一化
    class_rate = np.array(class_rate)
    class_rate /= np.sum(class_rate)
    shared_cov_mat = np.zeros((x_dim, x_dim))
    for i in range(class_num):
        shared_cov_mat += class_rate[i] * cov_mat[i]
    shared_cov_mat_pinv = np.linalg.pinv(shared_cov_mat)
    for i in range(class_num):
        w1.append(np.matmul(shared_cov_mat_pinv, mean_vec[i]))
        w2.append((-1 / 2) * np.matmul(np.matmul(mean_vec[i].T, shared_cov_mat_pinv), mean_vec[i]) + math.log(class_rate[i]))
    return w1, w2


def extract_test_dataset(test_x, test_y, class_num):
    new_test_x = None
    new_test_y = None
    for i in range(class_num):
        class_is_i_index_test = np.where(test_y == i)[0]
        class_is_i_x_test = np.array([test_x[i] for i in class_is_i_index_test]).T
        class_is_i_y_test = np.array([test_y[i] for i in class_is_i_index_test])
        if i == 0:
            new_test_x = class_is_i_x_test
            new_test_y = class_is_i_y_test
        else:
            new_test_x = np.hstack([new_test_x, class_is_i_x_test])
            new_test_y = np.append(new_test_y, class_is_i_y_test, axis=0)
    return new_test_x, new_test_y

def LDF(x, w1, w2):
    assert len(w1) == len(w2)
    class_num = len(w1)
    sample_num = x.shape[1]

    scores = np.zeros((class_num, sample_num))
    for i in range(class_num):
        score = np.matmul(w1[i].T, x) + w2[i]
        # print(np.matmul(w1[i].T, x), w2[i], score)
        scores[i,:] = score
    prediction = np.argmax(scores, axis=0)
    return prediction


def accuracy(ground_truth, prediction):
    return np.mean(np.equal(ground_truth, prediction))


if __name__ == "__main__":
    train_x, test_x, train_y, test_y = loadDataset()
    class_num = 2

    w1, w2 = calculate_weights(train_x, class_num)
    new_test_x, new_test_y = extract_test_dataset(test_x, test_y, class_num)

    prediction = LDF(new_test_x, w1, w2)

    acc = accuracy(new_test_y, prediction)
    print("Accuracy:", acc)