import keras
from keras.models import Sequential
from keras.layers import Dense
import random
import string
import numpy as np
import pickle

with open('noise_size_1000_testset.pickle', 'rb') as f:
    x_test, y_test, y_test_small = pickle.load(f)

def num2str(index1, index2, size=26):
    I2L = dict(zip(range(size), "ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    return ([I2L[index1], I2L[index2]])


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


def predict_results_only_2(model, x_train, y_train):
    predictions = model.predict(x_train)
    # index1, index2, label1, label2 represent two predictions and two labels respectively
    index1 = np.argmax(predictions[:, 0:26], axis=1)
    index2 = np.argmax(predictions[:, 26:52], axis=1)
    label1 = np.argmax(y_train[:, 0:26], axis=1)
    label2 = np.argmax(y_train[:, 26:52], axis=1)

    # Change number to string and do an output
    prediction_list = list(map(num2str, index1, index2))
    label_list = list(map(num2str, label1, label2))

    return (prediction_list, label_list)

def find_diff(list1, list2, ifprint = False):
    if "".join(list1) != "".join(list2):
        if ifprint == True:
            print("".join(list1), "".join(list2))
        return ("".join(list1), "".join(list2))
    else:
        return (True)

def misslabeled_data_genelization(sample_size = 2,loops = 1000, size = 26, key = 3, prob = 0.1, x_as_vector = False, y_as_vector = False):
    '''
        TODO: speed it up
    '''
    temp = ''.join([random.choice(string.ascii_uppercase),random.choice(string.ascii_uppercase)])

    ciphertext, ciphernum, plainnum = caeserde(temp[0:2], key = key, size = size, x_as_vector = x_as_vector, y_as_vector = y_as_vector)
    a = letter_position_matrix(plainnum, size = size)
    b = letter_position_matrix(ciphernum, size = size)
    # flatten label and training set
    label_equalsize = a.flatten()
    label_smaller = np.array(plainnum)
    train = b.flatten()
    for i in range(loops-1):
        # randomly generate strings
        temp = ''.join([random.choice(string.ascii_uppercase),random.choice(string.ascii_uppercase)])
        # caeser decoding
        ciphertext, ciphernum, plainnum = caeserde(temp, key = key, size = size, x_as_vector = x_as_vector, y_as_vector = y_as_vector)
        # get the letter position matrix for plaintext and cipertext
        b = letter_position_matrix(ciphernum, size = size)
        # sample randomly with prob = 0.1
        if random.random()<=prob:
            temp = ''.join([random.choice(string.ascii_uppercase),random.choice(string.ascii_uppercase)])
            ciphertext, ciphernum, plainnum = caeserde(temp, key = key, size = size, x_as_vector = x_as_vector, y_as_vector = y_as_vector)
        a = letter_position_matrix(plainnum, size = size)
        # get labels with 1 dimension (eg. abc, 123)
        label_smaller = np.vstack([label_smaller,np.array(plainnum)])
        # flatten label and training set
        label_equalsize = np.vstack([label_equalsize,a.flatten()])

        train = np.vstack([train,b.flatten()])
    return(train,label_equalsize,label_smaller)

def plot_acc_prob_curve_500(prob):
    x_train, y_train, y_train_small = misslabeled_data_genelization(loops=500, prob=prob)
    # assume that we have got our data x_train y_train
    # now we are going to train it in our model
    model = Sequential()
    model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    # change loss from possion to binary it works well :P
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train,y_train,epochs = 200, batch_size = 1000)
    
    prediction_list, label_list = predict_results_only_2(model, x_test, y_test)
    diff = list(map(find_diff, prediction_list, label_list))
    count = diff.count(True)/len(diff)
    return(count)

info500 = list(map(plot_acc_prob_curve_500,np.arange(0,1,.01)))

with open('noise_size_500.pickle', 'wb') as f:
    pickle.dump(info500, f)
