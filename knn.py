"""
Define a KNN class to classify document by document vector
"""
import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k, train_vectors, train_labels):
        self.k = k
        self.train_vectors = train_vectors
        self.train_labels = train_labels

    def label(self, test_vectors, cos=True):

        labels = []
        for i in range(test_vectors.shape[0]):
            dis = []
            k_labels = []
            for j in range(self.train_vectors.shape[0]):
                if cos:
                    dis.append(self.cosDis(test_vectors[i], self.train_vectors[j]))
                else:
                    dis.append(self.euclidDis(test_vectors[i], self.train_vectors[j]))
            if cos:
                order = np.argsort(-1 * np.array(dis))
            else:
                order = np.argsort(np.array(dis))
            #print(order)

            for m in range(self.k):
                #print(order[m])
                k_labels.append(self.train_labels[order[m]])
            label_count = Counter(k_labels)
            top_label = label_count.most_common(1)
            labels.append(top_label[0])
        return labels

    def scores(self, test_vectors, test_labels, cos=True):
        labels = self.label(test_vectors, cos)
        true_label_num = 0
        num = len(test_labels)

        for i in range(num):
            if test_labels[i] == labels[i][0]:
                true_label_num += 1
        #print(true_label_num)
        #print(test_labels)
        #print(labels)
        return true_label_num/num

    def euclidDis(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2)

    def cosDis(self, vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

