# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:40:37 2021

@author: tianh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../kaggle_data/'))
from utils import ESC50
import scipy
from scipy import signal
import IPython.display as ipd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import utils3 as U3
import random
import time

random.seed(2021)
np.random.seed(2021)

train_split = [[1,2,3],[1,2,4],[1,3,4],[2,3,4]]
holdout_split = [4,3,2,1]
test_split = [5,5,5,5]

def fetch_data(train_splits,holdout_splits,test_splits,category=[0,1,2,3,4,5]):
    
    shared_params = {'csv_path': '../kaggle_data/esc50.csv',
                     'wav_dir': '../kaggle_data/audio',
                     'dest_dir': '../kaggle_data/audio/16000',
                     'audio_rate': 16000,
                     'only_ESC10': False,
                     'pad': 0,
                     'normalize': True}
    
    train_size = 1200
    holdout_size = 400
    test_size = 400
    
    train_gen = ESC50(folds=train_splits,
                      randomize=False,
                      strongAugment=False,
                      random_crop=False,
                      inputLength=2,
                      mix=False,
                      **shared_params).batch_gen(train_size)
    
    holdout_gen = ESC50(folds=[holdout_splits],
                      randomize=False,
                      strongAugment=False,
                      random_crop=False,
                      inputLength=2,
                      mix=False,
                      **shared_params).batch_gen(holdout_size)
    
    test_gen = ESC50(folds=[test_split],
                      randomize=False,
                      strongAugment=False,
                      random_crop=False,
                      inputLength=4,
                      mix=False,
                      **shared_params).batch_gen(test_size)
    
    X_train, Y_train = next(train_gen)
    X_holdout, Y_holdout = next(holdout_gen)
    X_test, Y_test = next(test_gen)
    
    # pick types of sounds
    animal_sounds = category  # only train cat and dog
    mask_train = np.isin(Y_train,animal_sounds)
    mask_holdout = np.isin(Y_holdout,animal_sounds)
    mask_test = np.isin(Y_test,animal_sounds)
    X_train = X_train[mask_train]
    Y_train = Y_train[mask_train]
    X_holdout = X_holdout[mask_holdout]
    Y_holdout = Y_holdout[mask_holdout]
    X_test = X_test[mask_test]
    Y_test = Y_test[mask_test]
    train_size = len(Y_train)
    holdout_size = len(Y_holdout)
    test_size = len(Y_test)
    return X_train, Y_train, train_size, X_holdout, Y_holdout, holdout_size, X_test, Y_test, test_size


# plot some data points
# fig, axs = plt.subplots(4, 5, figsize=(13, 4))

# for idx in range(10):
#     i, j = int(idx / 5), int(idx % 5)
#     x = X_train[idx]
#     sampleFreqs, segmentTimes, sxx = signal.spectrogram(x[:, 0], 16000, window=('hann'))
#     axs[i*2][j].pcolormesh((len(segmentTimes) * segmentTimes / segmentTimes[-1]),
#                           sampleFreqs,
#                           10 * np.log10(sxx + 1e-15))
#     #axs[i*2][j].set_title(classes[seen_classes[idx]])
#     axs[i*2][j].set_axis_off()
#     axs[i*2+1][j].plot(x)
#     #axs[i*2+1][j].set_axis_off()
    
# plt.show()

def params_tune(X_train,Y_train,train_size,X_holdout,Y_holdout,holdout_size,truncate_size,C,gamma):
    # train samples
    hist_train, hist_labels_train, kmeans = U3.train_cluster(X_train,Y_train,train_size,truncate_size=truncate_size)

    # classification with SVM
    clf = SVC(kernel='rbf',C=C,gamma=gamma)  # default params
    clf.fit(hist_train, hist_labels_train)

    # test samples from fold 5
    hist_holdout, hist_labels_holdout = U3.predict_cluster(X_holdout,Y_holdout,holdout_size,kmeans,truncate_size=truncate_size)
    return U3.score_wld(clf,hist_holdout,hist_labels_holdout,holdout_size)

# start training
truncate_size = 5
# train samples
# hist_train, hist_labels_train, kmeans = U3.train_cluster(X_train,Y_train,train_size,truncate_size=truncate_size)

# classification with SVM
# clf = SVC(kernel='rbf')  # default params
# clf.fit(hist_train, hist_labels_train)

# # test samples from fold 5
# hist_holdout, hist_labels_holdout = U3.predict_cluster(X_holdout,Y_holdout,holdout_size,kmeans,truncate_size=truncate_size)
# print(U3.score_wld(clf,hist_holdout,hist_labels_holdout,holdout_size))
# #print(clf.score(hist_holdout,Y_holdout))

def cross_validation(train_split, holdout_split, test_split, truncate_size, C, gamma):
    results = 0
    for i in range(len(train_split)):
        X_train, Y_train, train_size, X_holdout, Y_holdout, holdout_size, X_test, Y_test, test_size = fetch_data(train_split[i],holdout_split[i],test_split[i])
        print(train_split[i])
        r = params_tune(X_train,Y_train,train_size,X_holdout,Y_holdout,holdout_size,truncate_size,C,gamma)
        print(r)
        results += r
    return results/(len(train_split))

X_train, Y_train, train_size, X_holdout, Y_holdout, holdout_size, X_test, Y_test, test_size = fetch_data(train_split[0],holdout_split[0],test_split[0])
C_range = 5*np.arange(-2,3,1)
gamma_range = 5*np.arange(-2,3,1)
results = np.ones((len(C_range),len(gamma_range)))
time_start = time.time()
for i in range(len(C_range)):
    for j in range(len(gamma_range)):
        C = np.power(10,C_range[i],dtype=float)
        gamma = np.power(10,gamma_range[j],dtype=float)
        # results[i,j] = (cross_validation(train_split, holdout_split, test_split, truncate_size, C, gamma))
        results[i,j] = params_tune(X_train,Y_train,train_size,X_holdout,Y_holdout,holdout_size,truncate_size,C,gamma)
        print((i*len(C_range)+j)/(len(C_range)*len(gamma_range)))
        print(time.time()-time_start)
        time_start = time.time()

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
