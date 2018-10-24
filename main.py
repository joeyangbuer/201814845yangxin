
from knn import KNN
import numpy as np
import pandas as pd
import csv


import time


start_time = time.time()
print("--start loading vectors...")
train_vectors = np.loadtxt(open(r"D:\python project\data\train vectors.csv", "r"), delimiter=",", skiprows=0)
test_vectors = np.loadtxt(open(r"D:\python project\data\test vectors.csv", "r"), delimiter=",", skiprows=0)
train_labels = pd.read_csv(r"D:\python project\data\train labels.csv", header=None, delimiter=",").values.tolist()
test_labels = pd.read_csv(r"D:\python project\data\test labels.csv", header=None, delimiter=",").values.tolist()
loading_time = time.time()
print("  loading vectors and labels end and cost %fs" % (loading_time-start_time))


for i in range(len(train_labels)):
    train_labels[i] = train_labels[i][0]
for i in range(len(test_labels)):
    test_labels[i] = test_labels[i][0]
#knn = KNN(3, train_vectors, train_labels)
#acc = knn.scores(test_vectors, test_labels)
#print("--calculating accuracy cost %fs , accuracy = %f" % (time.time()-loading_time, acc))


result_file = open(r"D:\python project\data\result.csv", "a")
writer = csv.writer(result_file)
for i in range(30, 31):
    knn = KNN(i, train_vectors, train_labels)
    acc = knn.scores(test_vectors, test_labels, False)
    writer.writerow([i, acc])
    print("k = %d, accuracy = %f" % (i, acc))
result_file.close()



