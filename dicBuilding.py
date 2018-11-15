"""
building a doc-word dictionary from documents
and record document frequency and word occurrence
"""
import math
from nltk.corpus import stopwords
from docParsing import DocParser
import numpy as np

stopwords = stopwords.words("english")

class DicBuilder:

    def __init__(self, doc_path_list, clean=False, delta=50, skip_word=stopwords):
        self.doc_path_list = doc_path_list
        self.doc_num = len(doc_path_list)
        self.doc_word_dic = self.build_dic(self.doc_path_list)
        self.all_word_occurrence = {}
        self.remains_word_occurrence = {}
        self.doc_freq = {}
        self.doc_len = dict(zip(self.doc_path_list, [0]*self.doc_num))

        for one_dic in self.doc_word_dic:
            word_count = self.doc_word_dic[one_dic]
            for item in word_count:
                self.doc_len[one_dic] += word_count[item]
                if item not in self.all_word_occurrence:
                    self.all_word_occurrence[item] = word_count[item]
                else:
                    self.all_word_occurrence[item] += word_count[item]

                if item not in self.doc_freq:
                    self.doc_freq[item] = 1
                else:
                    self.doc_freq[item] += 1
        if clean:
            for word in self.all_word_occurrence:
                if word not in skip_word and self.all_word_occurrence[word] >= delta:
                    self.remains_word_occurrence[word] = self.all_word_occurrence[word]
        else:
            self.remains_word_occurrence = self.all_word_occurrence
        '''
        with open("./data/all_word_occurrence.txt", "w") as f:
            for item in self.all_word_occurrence:
                f.write(item+"\t"+str(self.all_word_occurrence[item])+"\n")
        with open("./data/remains_word_occurrence.txt", "w") as f:
            for item in self.remains_word_occurrence:
                f.write(item+"\t"+str(self.remains_word_occurrence[item])+"\n")
        '''

    '''
    build doc-word dic
    '''
    def build_dic(self, doc_path_list):
        doc_word_dic = {}
        for one_doc_file in doc_path_list:
            parser = DocParser(one_doc_file)
            word_count = parser.parse()
            doc_word_dic[one_doc_file] = word_count
        return doc_word_dic
        #for item in self.doc_word_dic:
        #    print(item + "  " + str(self.doc_word_dic[item]))

    '''
    return a doc vector
    '''
    def vectorize(self, doc_word_dic):
        doc_vectors = []
        for doc in doc_word_dic:
            vector = []
            for word in self.remains_word_occurrence:
                if word not in doc_word_dic[doc]:
                    tf = 0
                else:
                    tf = 1 + math.log(doc_word_dic[doc][word], 10)

                idf = math.log(self.doc_num / self.doc_freq[word], 10)
                w = tf * idf
                vector.append(w)
            doc_vectors.append(vector)
        return np.array(doc_vectors)
