#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:22:25 2019

@author: xihajun
"""
from time import time
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

def dataprocess(datalist):
    for i in datalist:
        temp = []
        for j in range(26):
            if i!=j:
                temp.append(0)
            else:
                temp.append(1)
        try:
            transfered_data = np.vstack([transfered_data, temp])
        except:
            transfered_data = temp
    return(transfered_data)
    
def caeserde_26(plaintext, key=3):
    L2I = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", range(26)))
    I2L = dict(zip(range(26), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ciphertext = ""
    ciphernum = []
    plainnum = []

    for c in plaintext.upper():
        if c.isalpha():
            # L2I[c]+key represents the order eg A=1 B=2
            ciphertext += I2L[(L2I[c] + key) % 26]
            ciphernum.append((L2I[c] + key) % 26)
            plainnum.append((L2I[c]) % 26)
        else:
            ciphertext += c
            ciphernum.append('-1')
    ciphernum = dataprocess(ciphernum)
    #plainnum = dataprocess(plainnum)
    return (ciphertext, ciphernum, plainnum)



# create model
model = Sequential()
model.add(Dense(20, input_dim=26, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(26, activation='softmax'))



import keras
import keras.callbacks
from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir = "logs/test")
#tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# random data
text, train, label = caeserde_26(
    'WERTYUIOPOIUGFDSASXCVBNMKJHGFREWQASXCVBNMJKUJYTFRDCVBNMKLOIUYTFCVBABCDEFTYGUHIJGFTGHJKLGYFHJRTFYGUHIJOKIHUGYFTGHJKLJHJGFJKLJHGFDCVBNMHGFDSRTRYUHIJKOIUYTREWQAZXCVBNMFGHIJKLMNOPQRSTUVWXYZ'
)

import random
transfered_data = None
for i in label:
    temp = []
    k = random.randint(0,25)
    for j in range(26):
        if k!=j:
            temp.append(0)
        else:
            temp.append(1)
    try:
        transfered_data = np.vstack([transfered_data, temp])
    except:
        transfered_data = temp
    
    
transfered_data = None
#make sure the unique
emptylist = [[None for y in range(26)] for x in range(2)]
for i in label:
    k = 0
    if i in emptylist[0]:
        emptylist[0]
    else:
        emptylist[k] =  
        k+=1
    random.sample(range(0,25),25):
    temp = []
    for j in range(26):
        if i!=j:
            temp.append(0)
        else:
            temp.append(1)
    try:
        transfered_data = np.vstack([transfered_data, temp])
    except:
        transfered_data = temp
        
        
label = transfered_data

model.fit(train, label, epochs=500,callbacks=[tensorboard])
