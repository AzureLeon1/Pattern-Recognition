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


def multiclass_mse(data, labels):
    pinv_data = np.linalg.inv(data.T @ data + 1e-9 * np.eye(data.shape[1])) @ data.T
    a = pinv_data @ labels
    return a


if __name__ == "__main__":
    data = np.load('../datasets/LDFData/data.npz')['data']

    # p3
    train_data = np.vstack(data[:, :8, :])
    test_data = np.vstack(data[:, 8:, :])
    train_labels = np.repeat(np.eye(4), 8, axis=0)
    test_labels = np.repeat(np.eye(4), 2, axis=0)
    a = multiclass_mse(train_data, train_labels)
    # test
    scores = test_data @ a
    prediction = np.argmax(scores, axis=1)
    ground_truth = np.argmax(test_labels, axis=1)
    acc = (prediction == ground_truth).sum() / ground_truth.shape[0]
    print('MSE多类扩展方法准确率：{:.2%}'.format(acc))