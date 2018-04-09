# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:14:15 2018

@author: Home
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk.classify.util
from nltk.tokenize import word_tokenize

##----Support module. No runnable---------
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def clean_text(string):
    try:
        string = string.lower().replace("<br />", " ")
        cleaned = re.sub(strip_special_chars, "", string.lower())
        return cleaned
    except:
        print("Cannot clean text")
        return string.lower()
def bag_of_words(text):
    return text.lower().split(' ')

def pre_processing(corpus):
    res = []
    for text in corpus:
        text = text.lower()
        text = re.sub('[^a-zA-Z ]+','',text).strip()
        text = ' '.join(text.split())
        res.append(text)
    return res

def stopword_remove(text):
    bows = bag_of_words(text);
    filterd_stop_words = []
    stop_words = set(stopwords.words('english'))
    for word  in bows:
        if word not in stop_words:
            filtered_stop_words.append(word)
    text = ' '.join(filtered_stop_words)
    return text

def stemming(text):
    ps = PorterStemmer()
    bows = bag_of_words(text);
    stemmed = []
    for word in bows:
        stemmed.append(ps.stem(word))
    text = ' '.join(stemmed)
    return text

def document_term_matrix(corpus):
    tokenize = lambda doc: doc.lower().split(' ')
    tf_idf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    dtm = tf_idf.fit_transform(corpus)
    return dtm

def jaccard_word_index(text1,text2):
    text1 = set(text1.split(' '))
    text2 = set(text2.split(' '))
    return float(len(text1 & text2)) / len(text1 | text2)

def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict

def get_word_vector(text,tf_idf_term):
    word_vect = []
    words = bag_of_words(text)
    for term in tf_idf_term:
        term_count = 0
        for word in words:
            if word == term:
                term_count += 1
        word_vect.append(term_count)
    res = np.array(word_vect).reshape(1, -1)
    return res

def get_tf_idf_terms(df):
    corpus = df['text'].values
    tokenize = lambda doc: doc.lower().split(' ')
    tf_idf = TfidfVectorizer(max_features=2000,norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    tf_idf.fit_transform(corpus)
    feature_names = tf_idf.get_feature_names()
    return feature_names

def get_relevant_term(df,nterms=20):
    tokenize = lambda doc: doc.lower().split(' ')
    tf_idf = TfidfVectorizer(max_features=2000,norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    dtm = tf_idf.fit_transform(corpus)
    feature_names = tf_idf.get_feature_names()
    indices = np.argsort(tf_idf.idf_)[::-1]

    top_features = [feature_names[i] for i in indices[:nterms]]
    print(top_features)
    return top_features
