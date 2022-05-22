from joblib import Memory
from sklearn.datasets import load_svmlight_file
import time
mem = Memory('./mycache')

@mem.cache
def get_data():
    data = load_svmlight_file('/home/wangliang/datasets/Pattern-Recogniition/covtype_binary/covtype.libsvm.binary.scale')
    return data[0], data[1]

x, y = get_data()

# build the model
time_start = time.time()

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='sag', max_iter=200)
print('Train')
clf.fit(x, y)
time_end = time.time()
print('totally cost {} sec'.format(time_end - time_start))
print(clf.score(x, y))