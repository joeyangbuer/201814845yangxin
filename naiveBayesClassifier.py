from dicBuild import DicBuilder
import numpy as np
from nltk.corpus import stopwords
import time
import math
stopwords = stopwords.words("english")


def read_file(file):
    doc_path = []
    doc_class = []
    with open(file, "r") as f:
        for item in f:
            item = item.strip().split("\t")
            doc_path.append(item[0])
            doc_class.append(item[1])
    return doc_path, doc_class


read_time = time.time()
print("--start reading train/test data_preprocess path")
train_data_path, train_data_class = read_file(r"D:\python project\data\train.txt")
test_data_path, test_data_class = read_file(r"D:\python project\data\test.txt")
print("  end up reading train/test data_preprocess path and the time cost is %fs" % (time.time() - read_time))

build_time = time.time()
print("--start building dic and word frequency statistics")
train_dicBuilder = DicBuilder(train_data_path, train_data_class)
test_dicBuilder = DicBuilder(test_data_path, test_data_class)
class_word_dic, class_name, class_num, class_word_num = train_dicBuilder.count_word()
doc_word_dic = test_dicBuilder.build_dic()
print("  end up building dic and word frequency statistics and the time cost is %fs" % (time.time() - build_time))

score_time = time.time()
print("--start counting category scores")
score_matrix = []
for doc in doc_word_dic:
    word_dic = doc_word_dic[doc]
    class_score = []
    for one_class in class_name:
        score = math.log(class_num[one_class] / train_dicBuilder.doc_num, 10)
        for word in word_dic:
            if word in class_word_dic[one_class]:
                score += word_dic[word] * math.log(class_word_dic[one_class][word]/class_word_num[one_class], 10)
            else:
                score += word_dic[word] * math.log(1 / class_word_num[one_class], 10)
        class_score.append(score)
    score_matrix.append(class_score)
print("  end up counting category scores and the time cost is %fs" % (time.time() - score_time))

class_index = np.array(score_matrix).argmax(axis=1)
true_num = 0
for i in range(len(class_index)):
    if class_name[class_index[i]] == test_data_class[i]:
        true_num += 1
acc = true_num/len(test_data_path)
print("accuracy = %f" % acc)










