import numpy as np
import matplotlib.pyplot as plt


def extract_binary_class_data(data, positive_class, negative_class):
    assert positive_class in [1,2,3,4] and negative_class in [1,2,3,4]
    positive_data = data[positive_class-1]
    negative_data = data[negative_class-1]
    print('Shape of positive data:', positive_data.shape)
    print('Shape of negative data:', negative_data.shape)
    return positive_data, negative_data


def augment_norm(positive_data, negative_data):
    aug_pos_data = np.insert(positive_data, 0, np.ones(positive_data.shape[0]), axis=1)
    aug_neg_data = np.insert(negative_data, 0, np.ones(negative_data.shape[0]), axis=1)
    aug_neg_data = - aug_neg_data
    normed_data = np.vstack([aug_pos_data, aug_neg_data])
    return normed_data


def batch_perceptron(data):
    a = np.zeros([1, data.shape[1]])
    for epoch in range(1, 101):
        if epoch == 1:
            wrong_pred = np.where(a @ data.T == 0)[1]
        else:
            wrong_pred = np.where(a @ data.T < 0)[1]
        print('第%2d轮迭代，错分样本数：%2d' % (epoch, len(wrong_pred)))
        if len(wrong_pred) > 0:
            a += np.sum(data[wrong_pred], axis=0)
        else:
            break
    print('解向量：', a[0])
    print('收敛步数：', epoch-1)
    return a[0]


if __name__=="__main__":
    data = np.load('../datasets/LDFData/data.npz')['data']

    # p1-1
    # batch perception
    # positive class = 1, negative class = 2
    positive_class = 1
    negative_class = 2
    print('Positive class:', positive_class)
    print('Negative class:', negative_class)
    positive_data, negative_data = extract_binary_class_data(data, positive_class, negative_class)
    normed_data = augment_norm(positive_data, negative_data)
    a = batch_perceptron(normed_data)
    # visualization
    #判别面
    x = np.arange(-10, 10, 0.1)
    y = (- a[0] - a[1] * x) / a[2]
    plt.plot(x, y)
    #样本点
    class1 = plt.scatter(positive_data[:,0], positive_data[:,1], c='tomato', alpha=0.8)
    class2 = plt.scatter(negative_data[:,0], negative_data[:,1], c='mediumturquoise', alpha=0.8)
    plt.legend((class1, class2,), ("class 1", "class 2"), loc = 0)
    plt.title('batch-perception-w1-w2')
    plt.savefig("./img/batch-perception-w1-w2.png",dpi=500,bbox_inches = 'tight')
    plt.show()



    # p1-2
    # batch perception
    # positive class = 3, negative class = 2
    positive_class = 3
    negative_class = 2
    print('Positive class:', positive_class)
    print('Negative class:', negative_class)
    positive_data, negative_data = extract_binary_class_data(data, positive_class, negative_class)
    normed_data = augment_norm(positive_data, negative_data)
    a = batch_perceptron(normed_data)
    # visualization
    #判别面
    x = np.arange(-10, 10, 0.1)
    y = (- a[0] - a[1] * x) / a[2]
    plt.plot(x, y)
    #样本点
    class1 = plt.scatter(positive_data[:,0], positive_data[:,1], c='tomato', alpha=0.8)
    class2 = plt.scatter(negative_data[:,0], negative_data[:,1], c='mediumturquoise', alpha=0.8)
    plt.legend((class1, class2,), ("class 3", "class 2"), loc = 0)
    plt.title('batch-perception-w3-w2')
    plt.savefig("./img/batch-perception-w3-w2.png",dpi=500,bbox_inches = 'tight')
    plt.show()