# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:43:11 2018

@author: ThangPD
"""
import tensorflow as tf
from DataLoader import DataLoader
from datetime import datetime
import shutil
import os
import pickle
import numpy as np
import Text_Processor as tpc

class LSTM_Train(object):
    def __init__(self, learning_rate=0.001, epochs=20, batch_size=128, lstm_units=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.device_map = {'cpu':"/cpu:0", 'gpu':"/gpu:0"}

    def train(self, data, session_path=None, device_name='cpu'):
        if not isinstance (data,DataLoader):
            print("Invalid data")
            return

        self.data = data
        self.device_id = self.device_map.get(device_name,"/cpu:0")

        with tf.device(self.device_id):
            tf.reset_default_graph()

            X = tf.placeholder(tf.int32, [None, self.data.max_sentence_length], name = 'X')
            y = tf.placeholder(tf.float32, [None, self.data.n_classes], name = 'y')

            #embedded = tf.Variable(tf.zeros([batch_size, data.max_sentence_length, data.word_vector_len]),dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(self.data.word_vectors, X)

            #lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
            lstmCell = tf.nn.rnn_cell.LSTMCell(self.lstm_units)

            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
            value, _ = tf.nn.dynamic_rnn(lstmCell, embedded, dtype=tf.float32)

            weight = tf.Variable(tf.truncated_normal([self.lstm_units, self.data.n_classes]))
            bias = tf.Variable(tf.constant(0.1, shape=[self.data.n_classes]))
            value = tf.transpose(value, [1, 0, 2])
            last = tf.gather(value, int(value.get_shape()[0]) - 1)

            probability = tf.add(tf.matmul(last, weight), bias, name='probability')
            prediction = tf.argmax(probability,1, name = 'prediction')
            correctPred = tf.equal(prediction, tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=probability, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("----- START TRAINING. LEARNING RATE = {}, EPOCHS = {}, BATCH SIZE = {} -----".format(self.learning_rate, self.epochs, self.batch_size))
            for epoch in range(self.epochs):
                total_batch = len(self.data.train_sentence_arr)//self.batch_size
                epoch_cost = 0
                for batch_index in range(total_batch+1):
                    mini_batch_X, mini_batch_y = DataLoader.get_data_batch(batch_index, self.batch_size, self.data.train_sentence_arr, self.data.train_labels_arr)
                    _, c = sess.run([optimizer,cost], {X: mini_batch_X, y: mini_batch_y})
                    epoch_cost += c/total_batch
                    print(str(datetime.now().strftime("%Y-%b-%d %H:%M:%S"))+' - Epoch {}/{}, batch {}/{}'.format(epoch+1, self.epochs, batch_index, total_batch))

                train_accuracy = sess.run(accuracy, feed_dict={X: self.data.train_sentence_arr, y: self.data.train_labels_arr})
                test_accuracy = sess.run(accuracy, feed_dict={X: self.data.test_sentence_arr, y: self.data.test_labels_arr})
                print("---- Epoch {}/{}, cost = {:.3f}, Accuracy: train = {:.3f}, test = {:.3f}".format(epoch+1, self.epochs, epoch_cost, train_accuracy, test_accuracy))

            self.save_session(sess, session_path)
            print("--------------------------FINISH TRAIN MODEL---------------------------------")

    def save_session(self, session, session_path):
        if session_path==None: return
        if os.path.exists(session_path) and os.path.isdir(session_path):
            shutil.rmtree(session_path)
        os.makedirs(session_path)

        saver = tf.train.Saver(save_relative_paths=True)
        saver.save(session, os.path.join(session_path,'final_model'))

        train_info = {'word_list':self.data.word_list,
                      'word_vectors':self.data.word_vectors,
                      'labels_list':self.data.labels_list,
                      'n_classes':self.data.n_classes,
                      'max_sentence_length':self.data.max_sentence_length}

        with open(os.path.join(session_path,'train_info.pickle'), 'wb') as pkl:
            pickle.dump(train_info, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        print("Successfully save trained model. Check saved model at path: " + str(session_path))

class LSTM_Test(object):
    def __init__(self,session_path):
        self.session_path = session_path
        if not os.path.exists(session_path) or not os.path.isdir(session_path) or len(os.listdir(session_path))==0:
            print("Invalid session path")

        meta_files = [f for f in os.listdir(self.session_path) if f.endswith('.meta')]
        self.meta_path = os.path.join(self.session_path,meta_files[0])

        with open(os.path.join(self.session_path,'train_info.pickle'), 'rb') as pkl:
            train_info = pickle.load(pkl)
            self.word_list = train_info.get('word_list')
            self.word_vectors = train_info.get('word_vectors')
            self.labels_list = train_info.get('labels_list')
            self.n_classes = train_info.get('n_classes')
            self.max_sentence_length = train_info.get('max_sentence_length')


    def test(self,test_sample):
        if isinstance(test_sample,str):
            self.test_sentence(test_sample)
        if isinstance(test_sample,list):
            for s in test_sample:
                if isinstance(s,str):
                    self.test_sentence(s)

    def test_sentence(self,text):
        embed = self.embed_text(text)
        sentence = np.array([embed])
        label = np.array([np.zeros((self.n_classes,))])

        imported_meta = tf.train.import_meta_graph(self.meta_path)
        with tf.Session() as sess:
            imported_meta.restore(sess, tf.train.latest_checkpoint(self.session_path))

            graph = tf.get_default_graph()
            X = graph.get_tensor_by_name("X:0")
            y = graph.get_tensor_by_name("y:0")
            probability = graph.get_tensor_by_name("probability:0")
            prediction = graph.get_tensor_by_name("prediction:0")
            probs, preds = sess.run([probability, prediction], feed_dict={X: sentence, y: label})

            probs = list(probs[0])
            lbls = list(self.labels_list)
            res_dict = {x[0]:x[1] for x in zip(lbls,probs)}
            print("Sentence:",text)
            print("Sentiment probability:",res_dict)
            print("Predicted sentiment:",self.labels_list[preds[0]])

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
