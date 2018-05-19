import pandas as pd
import numpy as np
from delicious_loader import load_dataset
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ngram_range = 1
    maxlen = 200

    X_train, y_train, X_val, y_val, X_test, y_test, word_index = load_dataset(ngram_range=ngram_range, maxlen=maxlen)

    # Column 19 is the rarest

    # y_data = pd.read_csv('data/delicious/train-label.dat', header=None, sep=' ')
    # for col in range(0, 20, 1):
    #     print y_data[col].sum()

    X_test_unlabeled_pool = X_test[:1992, :]
    X_test_test = X_test[1992:, :]
    y_test_unlabeled_pool = y_test[:1992, -1]
    y_test_test = y_test[1992:, -1]

    acc = []
    dim = []

    for k in range(0,10,1):
        clf = LogisticRegression()
        print 'Size of X: ', len(X_train), X_train.shape, type(X_train)
        clf.fit(X_train, y_train[:, -1])

        preds = clf.decision_function(X_test_unlabeled_pool)

        values = []
        positions = []
        for i in range(0, len(X_test_unlabeled_pool), 1):
            values.append(abs(preds[i]))
            positions.append(i)

        for i in range(10):
            pos = np.array(values).argmin()
            # print np.array(values).min()
            X_train_new = np.zeros(((X_train.shape[0] + 1), X_train.shape[1]))
            y_train_new = np.zeros(((y_train[:, -1].shape[0] + 1), 1))

            X_train_new[:X_train.shape[0]] = X_train
            X_train_new[X_train.shape[0]:] = X_test_unlabeled_pool[pos, :]
            y_train_new[:y_train.shape[0],0] = y_train[:, -1]
            y_train_new[y_train.shape[0]:] = y_test_unlabeled_pool[pos]
            X_train = X_train_new
            y_train = y_train_new
            print len(X_train), y_train.shape
            np.delete(X_test_unlabeled_pool, pos, 0)
            np.delete(y_test_unlabeled_pool, pos, 0)
            del values[pos]
            del positions[pos]

        acc.append(clf.score(X_test_test, y_test_test))
        dim.append(X_train.shape[0])

    plt.plot(dim, acc)
    plt.show()



