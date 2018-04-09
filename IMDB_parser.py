# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:42:37 2018

@author: ThangPD
"""

import os
import numpy as np
import Text_Processor as tpc
import random

"""
Note: for faster implementation, copy 2 pre-processed files: word_list.npy and word_vectors.npy to the data folder.
For scratch implementation:
- word_list.npy: built by select all unique words from the corpus. TF-IDF can be used to select relevant words only.
- word_vectors.npy: simple one-hot-encode word vectors, or complex word embedded vector (using word2vec or Glove)
"""


class IMDB_parser(object):
    def __init__(self,data_path, parsed_path=None):
        self.data_path = data_path

        if parsed_path==None:
            parsed_path=data_path
        if not os.path.exists(parsed_path):
            os.makedirs(parsed_path)

        self.parsed_path = parsed_path

    def parse_data(self, max_sentence_length=250, max_file=None):
        self.max_sentence_length = max_sentence_length
        self.max_file = max_file
        self.word_list = np.load(os.path.join(self.data_path,'word_list.npy'))
        self.word_vectors = np.load(os.path.join(self.data_path,'word_vectors.npy'))

        self.n_classes = 2
        self.labels_list = np.array(['positive','negative'])
        np.save(os.path.join(self.parsed_path, 'labels_list.npy'), self.labels_list)

        train_pos_path = os.path.join(*[self.data_path, 'train', 'pos'])
        train_neg_path = os.path.join(*[self.data_path, 'train', 'neg'])
        test_post_path = os.path.join(*[self.data_path, 'test', 'pos'])
        test_neg_path = os.path.join(*[self.data_path, 'test', 'neg'])

        print("Start parsing data")
        train_pos_data = self.read_text_folder(train_pos_path)
        train_neg_data = self.read_text_folder(train_neg_path)
        test_pos_data = self.read_text_folder(test_post_path)
        test_neg_data = self.read_text_folder(test_neg_path)

        train_sentence_arr, train_labels_arr = self.merge_pos_neg_arr(train_pos_data, train_neg_data)
        test_sentence_arr, test_labels_arr = self.merge_pos_neg_arr(test_pos_data, test_neg_data)

        np.save(os.path.join(self.parsed_path, 'train_sentence_arr.npy'), train_sentence_arr)
        np.save(os.path.join(self.parsed_path, 'train_labels_arr.npy'), train_labels_arr)
        np.save(os.path.join(self.parsed_path, 'test_sentence_arr.npy'), test_sentence_arr)
        np.save(os.path.join(self.parsed_path, 'test_labels_arr.npy'), test_labels_arr)
        print("Successfully parsed data")

    def read_text_folder(self, folder_path):
        fnames = os.listdir(folder_path)
        if self.max_file is not None and isinstance(self.max_file,int) and self.max_file<=len(fnames):
            fnames = fnames[:self.max_file]

        fdata = np.zeros((len(fnames),self.max_sentence_length))
        for sentence_idx, fname in enumerate(fnames):
            print("Process file {}/{}".format(sentence_idx, len(fnames)))
            fpath = os.path.join(folder_path,fname)
            with open(fpath,'r',encoding='utf-8') as f:
                text = tpc.clean_text(f.read())
                split = text.split()
                for w_idx, word in enumerate(split):
                    idx = self.get_word_index(word)
                    if w_idx < self.max_sentence_length:
                        fdata[sentence_idx,w_idx] = idx
        return fdata

    def get_word_index(self, word):
        try:
            idx = self.word_list.index(word)
        except:
            #Unknown word
            idx = len(self.word_list)-1
        return idx

    def merge_pos_neg_arr(self, pos_arr, neg_arr, print_process=True):
        labels_arr = np.zeros((pos_arr.shape[0]+neg_arr.shape[0], self.n_classes))
        sentence_arr = np.zeros((pos_arr.shape[0]+pos_arr.shape[0], pos_arr.shape[1]))

        for i in range(len(sentence_arr)):
            if print_process==True: print("Processing arr line {}/{}".format(i,len(sentence_arr)))
            coin = random.random()
            if len(pos_arr)==0 and len(neg_arr)>0: coin=0.4
            if len(neg_arr)==0 and len(pos_arr)>0: coin=0.6
            if len(pos_arr)==0 and len(neg_arr)==0: break

            if coin > 0.5:
                idx = random.randint(0,len(pos_arr)-1)
                sentence = pos_arr[idx]
                sentence_arr[i]=sentence
                labels_arr[i]=np.array([1,0])
                pos_arr=np.delete(pos_arr,idx,0)
            else:
                idx = random.randint(0,len(neg_arr)-1)
                sentence = neg_arr[idx]
                sentence_arr[i]=sentence
                labels_arr[i]=np.array([0,1])
                neg_arr=np.delete(neg_arr,idx,0)

        return sentence_arr, labels_arr