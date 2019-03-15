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
import string
# fix random seed for reproducibility
np.random.seed(7)


# Define function to generate training set and lables, named x_train, y_train
# testing set named x_test, y_test


def dataprocess(datalist, data_size = 26):
    # function for 
    for i in datalist:
        temp = []
        for j in range(data_size):
            if i!=j:
                temp.append(0)
            else:
                temp.append(1)
        try:
            transfered_data = np.vstack([transfered_data, temp])
        except:
            transfered_data = temp
    return(transfered_data)
    
def caeserde(plaintext, key=3, size = 26, x_as_vector = True, y_as_vector = True):
    ''' plaintext: your original training data
        key = 3: shift key
        size = 26: alphabet size
        x_as_vector = Ture: means x_train is a vector with length = 26
        y_as_vector = Ture: the same
        '''
    L2I = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", range(size)))
    I2L = dict(zip(range(size), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ciphertext = ""
    ciphernum = []
    plainnum = []

    for c in plaintext.upper():
        if c.isalpha():
            # L2I[c]+key represents the order eg A=1 B=2
            ciphertext += I2L[(L2I[c] + key) % size]
            ciphernum.append((L2I[c] + key) % size)
            plainnum.append((L2I[c]) % size)
        else:
            ciphertext += c
            ciphernum.append('-1')
    if x_as_vector == True:
        ciphernum = dataprocess(ciphernum, size)
    if y_as_vector == True:
        plainnum = dataprocess(plainnum, size)
    return (ciphertext, ciphernum, plainnum)

def letter_position_matrix(text, key= 3, size = 26):
    ''' text can either be ciphertext or plaintext
    '''
    length = len(text)
    matrix = [[0 for y in range(size)] for x in range(length)]
    for idx, val in enumerate(text):
        matrix[idx][val] = 1
    matrix = np.array(matrix)
    return matrix

def data_genelization(sample_size=2,loops = 1000):
    temp = ''.join(random.choices(string.ascii_uppercase, k = sample_size))
    ciphertext, ciphernum, plainnum = caeserde_1(temp)
    a = letter_position_matrix(plainnum)
    b = letter_position_matrix(ciphernum)
    label_smaller = np.array(plainnum)
    # flatten label and training set
    label_equalsize = a.flatten()
    train = b.flatten()
    for i in range(loops-1):
        temp = ''.join(random.choices(string.ascii_uppercase, k = sample_size))
        ciphertext, ciphernum, plainnum = caeserde_1(temp)
        a = letter_position_matrix(plainnum)
        b = letter_position_matrix(ciphernum)
        label_smaller = np.vstack([label_smaller,np.array(plainnum)])
        # flatten label and training set
        label_equalsize = np.vstack([label_equalsize,a.flatten()])
        train = np.vstack([train,b.flatten()])
    return(train,label_equalsize,label_smaller)


def caeserde_train_26_label_26(plaintext, key=3):
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
    plainnum = dataprocess(plainnum)
    return (ciphertext, ciphernum, plainnum)


def caeserde_train_26_label_1(plaintext, key=3):
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

def caeserde_train_1_label_1(plaintext, key=3):
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
    #ciphernum = dataprocess(ciphernum)
    #plainnum = dataprocess(plainnum)
    return (ciphertext, ciphernum, plainnum)


def caeserde_train_1_label_26(plaintext, key=3):
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
    #ciphernum = dataprocess(ciphernum)
    plainnum = dataprocess(plainnum)
    return (ciphertext, ciphernum, plainnum)


def caeserde_simplify(plaintext, key=2, size = 3):
    L2I = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", range(size)))
    I2L = dict(zip(range(size), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ciphertext = ""
    ciphernum = []
    plainnum = []

    for c in plaintext.upper():
        if c.isalpha():
            # L2I[c]+key represents the order eg A=1 B=2
            ciphertext += I2L[(L2I[c] + key) % size]
            ciphernum.append((L2I[c] + key) % size)
            plainnum.append((L2I[c]) % size)
        else:
            ciphertext += c
            ciphernum.append('-1')
    ciphernum = dataprocess(ciphernum)
    plainnum = dataprocess(plainnum)
    return (ciphertext, ciphernum, plainnum)

