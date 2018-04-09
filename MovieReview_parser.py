# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:17:06 2018

@author: ThangPD
"""

import os
import numpy as np
import Text_Processor as tpc
import pandas as pd

"""
Note: for faster implementation, copy 2 pre-processed files: word_list.npy and word_vectors.npy to the data folder.
For scratch implementation:
- word_list.npy: built by select all unique words from the corpus. TF-IDF can be used to select relevant words only.
- word_vectors.npy: simple one-hot-encode word vectors, or complex word embedded vector (using word2vec or Glove)
"""

class MovieReview_parser(object):
    def __init__(self,data_path, parsed_path=None):
        self.data_path = data_path

        if parsed_path==None:
            parsed_path=data_path
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)

        self.parsed_path = parsed_path

    def parse_data(self, max_sentence_length=250):
        print("Start parsing data")
        self.max_sentence_length = max_sentence_length
        self.word_list = np.load(os.path.join(self.data_path,'word_list.npy'))
        self.word_vectors = np.load(os.path.join(self.data_path,'word_vectors.npy'))

        self.train_path = os.path.join(self.data_path,'train.tsv')
        self.test_path = os.path.join(self.data_path,'test.tsv')

        sentence_arr, labels_arr, self.labels_list = self.process_data_frame(self.train_path)

        train_test_ratio=0.2
        split_index = int(len(sentence_arr)*(1-train_test_ratio))

        self.train_sentence_arr = sentence_arr[:split_index]
        self.train_labels_arr = labels_arr[:split_index]
        self.test_sentence_arr = sentence_arr[split_index:]
        self.test_labels_arr = labels_arr[split_index:]

        np.save(os.path.join(self.parsed_path, 'labels_list.npy'), np.array(self.labels_list))
        np.save(os.path.join(self.parsed_path, 'train_sentence_arr.npy'), self.train_sentence_arr)
        np.save(os.path.join(self.parsed_path, 'train_labels_arr.npy'), self.train_labels_arr)
        np.save(os.path.join(self.parsed_path, 'test_sentence_arr.npy'), self.test_sentence_arr)
        np.save(os.path.join(self.parsed_path, 'test_labels_arr.npy'), self.test_labels_arr)
        print("Successfully parsed data")

    def process_data_frame(self,df_path):
        df = pd.read_csv(df_path,sep='\t')
        labels_list = list(df['Sentiment'].unique())

        df_tuple_list = list(df.groupby(['SentenceId']))
        sentente_dict_list = []
        for df_tuple in df_tuple_list:
            df = df_tuple[1]
            full_sentence = df.to_dict('records')[0]
            sentente_dict_list.append(full_sentence)

        sentence_arr = np.zeros((len(sentente_dict_list), self.max_sentence_length))
        labels_arr = np.zeros((len(sentente_dict_list), len(labels_list)))

        for idx, sentence_dic in enumerate(sentente_dict_list):
            sentence = sentence_dic.get('Phrase')
            lbl = sentence_dic.get('Sentiment')
            embed = self.embed_text(sentence)
            lbl_idx = labels_list.index(lbl)

            sentence_arr[idx]=embed
            labels_arr[idx][lbl_idx]=1

        return sentence_arr, labels_arr, labels_list

    def get_word_index(self, word):
        try:
            idx = self.word_list.index(word)
        except:
            #Unknown word
            idx = len(self.word_list)-1
        return idx

    def embed_text(self,text):
        embedded = np.zeros((self.max_sentence_length,))
        text = tpc.clean_text(text)
        words = text.lower().split()
        for idx,word in enumerate(words):
            word_idx = self.get_word_index(word)
            embedded[idx]=word_idx
        return embedded
