# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:17:06 2018

@author: ThangPD
"""

import sys
import os
from DataLoader import DataLoader
from IMDB_parser import IMDB_parser
from MovieReview_parser import MovieReview_parser
from LSTM_Classifier import LSTM_Train

def train_sentiment_classify(data_path,learning_rate=0.001, epochs=2, mini_batch_size=128, lstm_units=64, device_name='cpu'):
    data = DataLoader(data_path)
    if data.verify_data_folder_structure()==False:
        try:
            parser = IMDB_parser(data_path)
            parser.parse_data(max_sentence_length=250, max_file=1000)
        except:
            parser = MovieReview_parser(data_path)
            parser.parse_data(max_sentence_length=250)
        data.load_data()

    data.print_data_info()
    sample_sentence = '\"this is a simple sentence to test the embedded text\"'
    print("Embeded of:",sample_sentence," is:",data.embed_text(sample_sentence))

    clf = LSTM_Train(learning_rate=learning_rate, epochs=epochs, batch_size=mini_batch_size, lstm_units=lstm_units)
    clf.train(data, session_path=os.path.join(data_path,'saved_session'), device_name=device_name)


if __name__=='__main__':
    data_path = sys.argv[1]
    learning_rate = sys.argv[2]
    epochs = sys.argv[3]
    batch_size = sys.argv[4]
    lstm_units = sys.argv[5]
    device_name = sys.argv[6]
    train_sentiment_classify(data_path, learning_rate, epochs, batch_size, lstm_units, device_name)