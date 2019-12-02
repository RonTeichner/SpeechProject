# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:26:59 2015

@author: Abhijeet Kumar
@code :  This program implemets feature (MFCC + delta)
         extraction process for an audio. 
@Note :  20 dim MFCC(19 mfcc coeff + 1 frame log energy)
         20 dim delta computation on MFCC features. 
@output : It returns 40 dimensional feature vectors for an audio.
"""

import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

def extract_features(audio,rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""    
    nfft = int(np.exp2(np.ceil(np.log2(audio.shape[0]))))
    mfcc_feat = mfcc.mfcc(signal=audio, samplerate=rate, winlen=0.025, winstep=0.01, numcep=13, nfft=nfft, appendEnergy=True)
    #mfcc_feat = preprocessing.scale(mfcc_feat)

    mfcc_delta = mfcc.base.delta(mfcc_feat,1)
    mfcc_doubleDelta = mfcc.base.delta(mfcc_feat, 2)


    combined = np.hstack((mfcc_feat, mfcc_delta, mfcc_doubleDelta))
    return combined
#    
if __name__ == "__main__":
     print("In main, Call extract_features(audio,signal_rate) as parameters")
     
