# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 05:25:57 2021

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
import random


def train_cluster(X,Y,batch_size,truncate_size=20,n_clusters=8, sample_rate=16000, time_window=1):
    random.seed(2021)
    np.random.seed(2021)

    stft = None
    for i in range(batch_size):
        x = X[i]
        sampleFreqs, segmentTimes, sxx = signal.spectrogram(x[:, 0], sample_rate, window=('hann'))
        if stft is None:
            stft = np.ndarray((batch_size,)+sxx.shape)
            stft[0] = sxx
        else:
            stft[i] = sxx
            
    # SVD
    U = None
    S = None
    Vh = None
    for i in range(batch_size):
        u, s, vh = scipy.linalg.svd(stft[i])
        if U is None:
            U = np.ndarray((batch_size,)+u.shape)
            U[0] = u
        else:
            U[i] = u
        if S is None:
            S = np.ndarray((batch_size,)+s.shape)
            S[0] = s
        else:
            S[i] = s
        if Vh is None:
            Vh = np.ndarray((batch_size,)+vh.shape)
            Vh[0] = vh
        else:
            Vh[i] = vh
            
    # Truncate columns of Vh
    Vh_trunc = Vh[:, 0:truncate_size, :]
    # reshape Vh
    time_bins = Vh_trunc.shape[2]
    sample_number = Vh_trunc.shape[0]
    total_time_bins = time_bins*sample_number
    Vh_trunc_reshape = np.ndarray((total_time_bins, truncate_size))
    # y index if needed
    for i in range(total_time_bins):
        idx0 = i//time_bins
        idx2 = i%time_bins
        Vh_trunc_reshape[i,:] = Vh_trunc[idx0,:,idx2]
    
    # K means clustering

    kmeans = KMeans(n_clusters=n_clusters).fit(Vh_trunc_reshape)
    
    # split into time windows
    number_unit_window = len(X[0])//(sample_rate*time_window)

    point_unit_window = len(segmentTimes)//number_unit_window

    # histogram of each samples
    hist_count = np.zeros((batch_size*number_unit_window, n_clusters))
    for i in range(total_time_bins):
        idx0 = i//time_bins
        idx1 = i%time_bins
        idx3 = idx1//point_unit_window
        if idx3 <= 4:
            idx2 = 5*idx0 + idx3
            for k in range(n_clusters):
                if kmeans.labels_[i]==k:
                    hist_count[idx2,k] = hist_count[idx2,k]+1
    hist = hist_count/(point_unit_window)
    # create labels for hist
    hist_labels = np.empty(number_unit_window*batch_size)
    for i in range(len(hist_labels)):
        hist_labels[i] = Y[i//number_unit_window]
        
    return hist, hist_labels, kmeans

def predict_cluster (X,Y,batch_size,kmeans,truncate_size=20,n_clusters=8, sample_rate=16000, time_window=1):
    random.seed(2021)
    np.random.seed(2021)

    stft = None
    for i in range(batch_size):
        x = X[i]
        sampleFreqs, segmentTimes, sxx = signal.spectrogram(x[:, 0], sample_rate, window=('hann'))
        if stft is None:
            stft = np.ndarray((batch_size,)+sxx.shape)
            stft[0] = sxx
        else:
            stft[i] = sxx
            
    # SVD
    U = None
    S = None
    Vh = None
    for i in range(batch_size):
        u, s, vh = scipy.linalg.svd(stft[i])
        if U is None:
            U = np.ndarray((batch_size,)+u.shape)
            U[0] = u
        else:
            U[i] = u
        if S is None:
            S = np.ndarray((batch_size,)+s.shape)
            S[0] = s
        else:
            S[i] = s
        if Vh is None:
            Vh = np.ndarray((batch_size,)+vh.shape)
            Vh[0] = vh
        else:
            Vh[i] = vh
            
    # Truncate columns of Vh
    Vh_trunc = Vh[:, 0:truncate_size, :]
    # reshape Vh
    time_bins = Vh_trunc.shape[2]
    sample_number = Vh_trunc.shape[0]
    total_time_bins = time_bins*sample_number
    Vh_trunc_reshape = np.ndarray((total_time_bins, truncate_size))
    # y index if needed
    for i in range(total_time_bins):
        idx0 = i//time_bins
        idx2 = i%time_bins
        Vh_trunc_reshape[i,:] = Vh_trunc[idx0,:,idx2]
    
    # K means clustering predict
    labels = kmeans.predict(Vh_trunc_reshape)
    
    
    # split into time windows
    number_unit_window = len(X[0])//(sample_rate*time_window)
    point_unit_window = len(segmentTimes)//number_unit_window
    # histogram of each samples
    hist_count = np.zeros((batch_size*number_unit_window, n_clusters))
    for i in range(total_time_bins):
        idx0 = i//time_bins
        idx1 = i%time_bins
        idx3 = idx1//point_unit_window
        if idx3 <= 4:
            idx2 = 5*idx0 + idx3
            for k in range(n_clusters):
                if labels[i]==k:
                    hist_count[idx2,k] = hist_count[idx2,k]+1
    hist = hist_count/(point_unit_window)
    # create labels for hist
    hist_labels = np.empty(number_unit_window*batch_size)
    for i in range(len(hist_labels)):
        hist_labels[i] = Y[i//number_unit_window]
        
    return hist, hist_labels


def score_wld (clf,holdout,holdout_labels,holdout_size):
    pred = clf.predict(holdout)
    count = 0
    correct = 0
    for i in np.split((pred==holdout_labels),holdout_size):
        count+=1
        if np.sum(i)>=3:
            correct+=1
    return correct/count
