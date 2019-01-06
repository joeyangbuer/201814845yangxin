"""
building a doc-word dictionary from documents
and record document frequency and word occurrence
"""
import math
from nltk.corpus import stopwords
from docParse import DocParser
import numpy as np

stopwords = stopwords.words("english")

class DicBuilder:

    def __init__(self, doc_path_list, doc_label_list=None):
        self.doc_path_list = doc_path_list
        self.doc_num = len(doc_path_list)
        self.doc_label_list = doc_label_list

        '''
        with open("./data_preprocess/all_word_occurrence.txt", "w") as f:
            for item in self.all_word_occurrence:
                f.write(item+"\t"+str(self.all_word_occurrence[item])+"\n")
        with open("./data_preprocess/remains_word_occurrence.txt", "w") as f:
            for item in self.remains_word_occurrence:
                f.write(item+"\t"+str(self.remains_word_occurrence[item])+"\n")
        '''

    '''
    build doc-word dic
    '''
    def build_dic(self):
        doc_word_dic = {}
        for one_doc_file in self.doc_path_list:
            parser = DocParser(one_doc_file)
            word_count = parser.parse()
            doc_word_dic[one_doc_file] = word_count
        return doc_word_dic
        #for item in self.doc_word_dic:
        #    print(item + "  " + str(self.doc_word_dic[item]))

    def parse_dic(self, clean=False, delta=50, skip_word=stopwords):
        doc_word_dic = self.build_dic()
        all_word_occurrence = {}
        remains_word_occurrence = {}
        doc_freq = {}
        doc_len = dict(zip(self.doc_path_list, [0] * self.doc_num))

        for one_dic in doc_word_dic:
            word_count = doc_word_dic[one_dic]
            for item in word_count:
                doc_len[one_dic] += word_count[item]
                if item not in all_word_occurrence:
                    all_word_occurrence[item] = word_count[item]
                else:
                    all_word_occurrence[item] += word_count[item]

                if item not in doc_freq:
                    doc_freq[item] = 1
                else:
                    doc_freq[item] += 1
        if clean:
            for word in all_word_occurrence:
                if word not in skip_word and all_word_occurrence[word] >= delta:
                    remains_word_occurrence[word] = all_word_occurrence[word]
        else:
            remains_word_occurrence = all_word_occurrence

        idf_weight = {}
        for word in remains_word_occurrence:
            idf = math.log(self.doc_num / doc_freq[word], 10)
            idf_weight[word] = idf

        return doc_word_dic, all_word_occurrence, remains_word_occurrence, doc_freq, idf_weight

    def count_word(self):
        doc_word_dic = self.build_dic()
        class_word_dic = {}
        class_num = {}
        class_word_num = {}
        class_name = []
        for (doc, label) in zip(doc_word_dic, self.doc_label_list):
            word_dic = doc_word_dic[doc]
            if label not in class_word_dic:
                class_word_dic[label] = word_dic
            else:
                for word in word_dic:
                    if word not in class_word_dic[label]:
                        class_word_dic[label][word] = word_dic[word]
                    else:
                        class_word_dic[label][word] += word_dic[word]

            if label not in class_name:
                class_name.append(label)

            if label not in class_num:
                class_num[label] = 1
            else:
                class_num[label] += 1

            for word in word_dic:
                if label not in class_word_num:
                    class_word_num[label] = word_dic[word]
                else:
                    class_word_num[label] += word_dic[word]

        return class_word_dic, class_name, class_num, class_word_num

