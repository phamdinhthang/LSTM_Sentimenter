# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:49:14 2018

@author: ThangPD

----------------Data Loader module for Recurrent Neural Network (LTSM) ----------------------------------
For different datasets, a dedicated loader module is neccessary to convert the raw format of the dataset to the format that DataLoader can read. The data folder that DataLoader can read have structure decribed below


Data Folder must contain the following files:

1. word_list.npy:
    1D numpy array, shape = (n_words,) , store all the words in the vocabulary, in lowercase.

2. word_vectors.npy:
    2D numpy array, shape = (n_words, word_embedded_length) . The order of n_words must be identical to the word_list.npy. The word_embedded_length is the len of embedded vector for each word. Embedding method can be simpy one-hot-encoding or complex word-embedding (via word2vec)

3. labels_list.npy
    1D numpy array, shape = (n_classes,) , store all possible labels for the dataset.
    For example: array(['very_positive', 'positive', 'neutral', 'negative', 'very_negative'])

3. train_sentence_arr.npy
    2D numpy array, shape = (n_samples, max_sentence_length) . Each sentence shall be encoded into a 1D array of shape (n_words,) . To unify the data shape, every sentence is of length max_sentence_length. Shorter sentence shall be zero padded at the end. Longer sentence shall be clipped at the end. Each element of the 1D array is the index of the corresponding word in the vocabulary file (word_list.npy)
    For example: "hello world from python" shall be encoded into array([10, 600, 40, 50, 0, 0, 0,...])

    Later, the model shall look up the train_sentence_arr from the word_vectors, to create an embedded matrix for each sentence, with shape = (max_sentence_length, word_embedded_length). The overall train input shall be of shape (n_samples, max_sentence_length, word_embedded_length)

4. train_labels_arr.npy:
    2D numpy array, shape = (n_samples, n_classes) . Labels for the train_sentence_arr. The order of n_samples must be identical to the train_sentence_arr.npy. The order of n_classes is one-hot-encoded using the labels_list.npy

5. test_sentence_arr.npy:
    Similar to train_sentence_arr.npy, use to test the model

6. test_labels_arr.npy:
    Similar to train_labels_arr.npy, use to test the model
"""

import os
import numpy as np
import Text_Processor as tpc

class DataLoader(object):
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.load_data()

    def verify_data_folder_structure(self):
        self.word_list_path = os.path.join(self.data_folder_path, 'word_list.npy')
        self.word_vectors_path = os.path.join(self.data_folder_path, 'word_vectors.npy')
        self.lbl_list_path = os.path.join(self.data_folder_path, 'labels_list.npy')
        self.train_stc_path = os.path.join(self.data_folder_path, 'train_sentence_arr.npy')
        self.train_lbl_path = os.path.join(self.data_folder_path, 'train_labels_arr.npy')
        self.test_stc_path = os.path.join(self.data_folder_path, 'test_sentence_arr.npy')
        self.test_lbl_path = os.path.join(self.data_folder_path, 'test_labels_arr.npy')

        fpaths = [self.lbl_list_path, self.train_stc_path, self.train_lbl_path, self.test_stc_path, self.test_lbl_path]
        for path in fpaths:
            if not os.path.exists(path) or not os.path.isfile(path):
                print("Invalid path:",path)
                return False
        return True

    def load_data(self):
        if self.verify_data_folder_structure()==False:
            print("Invalid data folder structure. Please verify datafolder structure")
            return

        self.labels_list = np.load(self.lbl_list_path)
        self.n_classes = len(self.labels_list)

        self.word_list = np.load(self.word_list_path)
        if not isinstance(self.word_list[0],str):
            self.word_list = [word.decode('UTF-8') for word in self.word_list]

        self.word_vectors = np.load(self.word_vectors_path)
        self.train_sentence_arr = np.load(self.train_stc_path)
        self.train_labels_arr = np.load(self.train_lbl_path)
        self.test_sentence_arr = np.load(self.test_stc_path)
        self.test_labels_arr = np.load(self.test_lbl_path)
        self.max_sentence_length = self.train_sentence_arr.shape[1]

    def print_data_info(self):
        print("----------- Data Folder information -----------")
        print("--- Number of classes: {}".format(self.n_classes))
        print("--- Classes:",self.labels_list)
        print("--- Vocabulary size: {} words".format(len(self.word_list)))
        print("--- Word vectors shape:",self.word_vectors.shape)
        print("--- Train data shape:", self.train_sentence_arr.shape)
        print("--- Train label shape:", self.train_labels_arr.shape)
        print("--- Test data shape:", self.test_sentence_arr.shape)
        print("--- Test label shape:", self.test_labels_arr.shape)
        print("--- Max sentence length:",self.max_sentence_length)

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

    @staticmethod
    def get_data_batch(batch_index, batch_size, X, y):
        if len(X) != len(y):
            print("Different X and y shape")
            return None, None
        data_size = len(X)

        if batch_index*batch_size > data_size:
            print("Invalid batch index and/or batch size")
            return None, None

        start_idx = batch_index*batch_size
        end_idx = start_idx+batch_size
        if end_idx>(data_size-1): end_idx=data_size-1

        return X[start_idx:end_idx,:], y[start_idx:end_idx,:]