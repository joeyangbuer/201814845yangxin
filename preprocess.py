from dicBuliding import DicBuilder
import numpy as np
import pandas as pd
import time


def read_file(file):
    doc_path = []
    doc_class = []
    with open(file, "r") as f:
        for item in f:
            item = item.strip().split("\t")
            doc_path.append(item[0])
            doc_class.append(item[1])
    return doc_path, doc_class


start_time = time.time()
print("--start reading train/test data path")
train_data_path, train_data_class = read_file(r"D:\python project\data\train.txt")
test_data_path, test_data_class = read_file(r"D:\python project\data\test.txt")
print("  end up reading train/test data path")

print("--start building dic")
dicBuilder = DicBuilder(train_data_path, True, 50)
build_dic_time = time.time()
print("  end up building dic and the time cost is %fs" % (build_dic_time-start_time))

print("--start document vectorization and the dimension is %d" % len(dicBuilder.remains_word_occurrence))
train_vectors = dicBuilder.vectorize(dicBuilder.doc_word_dic)
test_vectors = dicBuilder.vectorize(dicBuilder.build_dic(test_data_path))
print("  vectorization end and the time cost is: %fs" % (time.time()-build_dic_time))

print("--save the document vector and document category in the csv file")
np.savetxt(r"D:\python project\data\train vectors.csv", train_vectors, delimiter=",")
np.savetxt(r"D:\python project\data\test vectors.csv", test_vectors, delimiter=",")
df = pd.DataFrame(train_data_class)
df.to_csv(r"D:\python project\data\train labels.csv", index=False, header=False)
df = pd.DataFrame(test_data_class)
df.to_csv(r"D:\python project\data\test labels.csv", index=False, header=False)