#coding:utf-8
"""
pre-processing one document and return a dic including word-frequency pair
(return {word1:freq1, word2,freq2....})
"""

from textblob import TextBlob
import re


class DocParser:
    """
    read a document
    """
    def __init__(self, doc_path):
        self.doc_path = doc_path
        with open(doc_path,  "r", encoding="latin-1") as self.file:
            self.doc = self.file.read()

    '''
    return a dic including word-frequency pair
    '''
    def parse(self):
        doc = self.doc.lower()
        doc = re.sub("[^A-Za-z\\s]", "", doc)
        parser = TextBlob(doc)
        #single_word = parser.words.singularize()
        word_count = parser.word_counts
        #print(single_word)
        #print(word_count)
        return word_count
