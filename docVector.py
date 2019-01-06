"""
Construct a vector for a doc with tf-idf weight
and store them
"""

from dicBuild import DicBuilder
import numpy as np
import pandas as pd
import time
import math

#read doc paths
def read_file(file):
    doc_path = []
    doc_class = []
    with open(file, "r") as f:
        for item in f:
            item = item.strip().split("\t")
            doc_path.append(item[0])
            doc_class.append(item[1])
    return doc_path, doc_class

#construct a vector for a doc
def vectorize(doc_word_dic, remains_word_occurrence, idf_weight):
    doc_vectors = []
    for doc in doc_word_dic:
        vector = []
        for word in remains_word_occurrence:
            if word not in doc_word_dic[doc]:
                tf = 0
            else:
                tf = 1 + math.log(doc_word_dic[doc][word], 10)

            idf = idf_weight[word]
            w = tf * idf
            vector.append(w)
        doc_vectors.append(vector)
    return np.array(doc_vectors)


start_time = time.time()
print("--start reading train/test data_preprocess path")
train_data_path, train_data_class = read_file(r"D:\python project\data\train.txt")
test_data_path, test_data_class = read_file(r"D:\python project\data\test.txt")
print("  end up reading train/test data_preprocess path")

print("--start building dic")
train_dicBuilder = DicBuilder(train_data_path)
doc_word_dic, all_word_occurrence, remains_word_occurrence, doc_freq, idf_weight = train_dicBuilder.parse_dic(True, 50)
build_dic_time = time.time()
print("  end up building dic and the time cost is %fs" % (build_dic_time-start_time))

print("--start document vectorization and the dimension is %d" % len(remains_word_occurrence))
train_vectors = vectorize(doc_word_dic, remains_word_occurrence, idf_weight)

test_dicBuilder = DicBuilder(test_data_path)
doc_word_dic = test_dicBuilder.build_dic()
test_vectors = vectorize(doc_word_dic, remains_word_occurrence, idf_weight)
print("  vectorization end and the time cost is: %fs" % (time.time()-build_dic_time))

print("--save the document vector and document category in the csv file")
np.savetxt(r"D:\python project\data\train vectors.csv", train_vectors, delimiter=",")
np.savetxt(r"D:\python project\data\test vectors.csv", test_vectors, delimiter=",")
df = pd.DataFrame(train_data_class)
df.to_csv(r"D:\python project\data\train labels.csv", index=False, header=False)
df = pd.DataFrame(test_data_class)
df.to_csv(r"D:\python project\data\test labels.csv", index=False, header=False)