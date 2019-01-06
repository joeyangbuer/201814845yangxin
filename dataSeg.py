"""
segmenting dataset to train set and test set according to average sampling for each category
then recording the document path separately
"""

import os
import random

docs_file = r"D:\python project\data\documents"
train_file = r"D:\python project\data\train.txt"
test_file = r"D:\python project\data\test.txt"
train_ratio = 0.8
test_ratio = 1-train_ratio
train_data = []
test_data = []

print("start data_preprocess segmentation, train data_preprocess size : test data_preprocess size = 8:2")

for class_name in os.listdir(docs_file):
    class_file = os.path.join(docs_file,  class_name)
    doc_list = os.listdir(class_file)

    random.shuffle(doc_list)
    train_num = int(len(doc_list)*train_ratio)

    for i in range(train_num):
        train_data.append((os.path.join(class_file, doc_list[i]), class_name))

    for i in range(train_num, len(doc_list)):
        test_data.append((os.path.join(class_file, doc_list[i]), class_name))

# shuffle the order of data_preprocess
random.shuffle(train_data)
random.shuffle(test_data)

print("write down the train_data_path and test_data_path")

# write down the division
with open(train_file,"w") as f:
    for item in train_data:
        f.write(item[0]+"\t"+item[1]+"\n")
        # f.write(item[0] + "\n")

with open(test_file,"w") as f:
    for item in test_data:
        f.write(item[0]+"\t"+item[1]+"\n")
        # f.write(item[0] + "\n")

print("data_preprocess segmentation end")
