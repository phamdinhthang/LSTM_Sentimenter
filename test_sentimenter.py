# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 18:19:14 2018

@author: ThangPD
"""

import sys
from LSTM_Classifier import LSTM_Test

def test_sentiment_classify(session_path, sentence):
    clf = LSTM_Test(session_path)
    clf.test(sentence)

if __name__=='__main__':
    session_path = sys.argv[1]
    sentence = sys.argv[2]
    test_sentiment_classify(session_path, sentence)