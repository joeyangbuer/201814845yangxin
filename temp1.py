import re
import numpy as np
from collections import Counter
from textblob import TextBlob

'''
k = 3
train_labels = ["a", "b", "c", "a", "b", "b", "a", "b"]
dis = [0.005, 0.3, 0.6, 0.7, 0.8, 0.2, 0.1, 0.05]
k_labels = []
labels = []
order = np.argsort(np.array(dis))

for m in range(k):
        k_labels.append(train_labels[order[m]])
label_count = Counter(k_labels)
top_label = label_count.most_common(1)
labels.append(top_label[0])

print(label_count)
print(labels)



def cosDis(vector1, vector2):
    print(vector1.T.shape)
    print(np.dot(vector1, vector2))
    print(np.linalg.norm(vector2))
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

print(cosDis(np.array([0,1,1]),np.array([0,0,1])))
print(["2"]+["344444","sss"])'''


a = np.array([[1,2,3], [4,2,1]])
print(a.argmax(axis=1))