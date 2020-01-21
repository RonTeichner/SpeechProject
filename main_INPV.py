import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from audioMNIST_mata2Dict import *
from speakerfeatures import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
import random
import os
import pickle
import time
from hmmlearn.hmm import GMMHMM as GMMHMM
from sklearn.mixture import GaussianMixture
import sounddevice as sd
# Import the package and create an audio effects chain function.
from pysndfx import AudioEffectsChain
import json
import data as data_module
import net as net_module

# random.seed(0)

path2FigTrain = './SentencesFigTrain.png'
path2FigValidate = './SentencesFigValidate.png'
path2FigTest = './SentencesFigTest.png'

path2sentencesAudioInputMatrixTrain = './sentencesAudioInputMatrixTrain.pt'
path2sentencesAudioInputMatrixValidate = './sentencesAudioInputMatrixValidate.pt'
path2sentencesAudioInputMatrixTest = './sentencesAudioInputMatrixTest.pt'

path2SentencesResultsTrain = './SentencesResultsTrain.pt'
path2SentencesResultsValidate = './SentencesResultsValidate.pt'
path2SentencesResultsTest = './SentencesResultsTest.pt'

sentencesAudioInputMatrixValidate = pickle.load(open(path2sentencesAudioInputMatrixValidate, "rb"))
sentencesAudioInputMatrixTest = pickle.load(open(path2sentencesAudioInputMatrixTest, "rb"))
sentencesAudioInputMatrixTrain = pickle.load(open(path2sentencesAudioInputMatrixTrain, "rb"))

sentencesEstimationResultsTrain = pickle.load(open(path2SentencesResultsTrain, "rb"))
sentencesEstimationResultsValidate = pickle.load(open(path2SentencesResultsValidate, "rb"))
sentencesEstimationResultsTest = pickle.load(open(path2SentencesResultsTest, "rb"))

tsf_name = 'AudioTransforms'
tsf_args = {'channels': 'avg', 'noise': [0.3, 0.001], 'crop': [0.4, 0.25]}
t_transforms = getattr(data_module, tsf_name)('train', tsf_args)
v_transforms = getattr(data_module, tsf_name)('val', tsf_args)
print(t_transforms)

nTimeIndexes = 5
beta = 0
m_name = 'AudioCRNN'

config = json.load(open('my-config.json'))
config['net_mode'] = 'init'
config['cfg'] = 'crnn.cfg'
model = net_module.AudioCRNN(config=config).to('cuda')

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

opt_name = config['optimizer']['type']
opt_args = config['optimizer']['args']
#opt_args['lr'] = 1e-3
optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)

lr_name = config['lr_scheduler']['type']
lr_args = config['lr_scheduler']['args']
#lr_args['step_size'] = 1000
if lr_name == 'None':
    lr_scheduler = None
else:
    lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)

wordsGroundTruthList = pickle.load(open('./wordsGroundTruth.pt', "rb"))
wordsTrain, wordsValidate, wordsTest = wordsGroundTruthList[0], wordsGroundTruthList[1], wordsGroundTruthList[2]

nEpochs = 1
trainLoss = np.zeros(nEpochs)
epochVecTrain = np.arange(nEpochs)
validateLoss, testLoss, epochVec = list(), list(), list()
for epochIdx in range(nEpochs):
    trainLoss[epochIdx], _ = trainFunc(t_transforms, v_transforms, sentencesAudioInputMatrixTrain, wordsTrain, model, optimizer, lr_scheduler, epochIdx, validateOnly=False)

    if True:  # epochIdx % 10 == 0 and epochIdx > 0:
        epochVec.append(epochIdx)

        _, probabilitiesLUT = trainFunc(t_transforms, v_transforms, sentencesAudioInputMatrixTrain[:1000], wordsTrain[:1000], model, optimizer, lr_scheduler, epochIdx, validateOnly=True)
        plotSentenceResults('Train', sentencesEstimationResultsTrain[:1000], path2FigValidate.split('.png')[0] + '_Trainepoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        validateLossEpoch, probabilitiesLUT = trainFunc(t_transforms, v_transforms, sentencesAudioInputMatrixValidate, wordsValidate, model, optimizer, lr_scheduler, epochIdx, validateOnly=True)
        validateLoss.append(validateLossEpoch)
        plotSentenceResults('Validate', sentencesEstimationResultsValidate, path2FigValidate.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        testLossEpoch, probabilitiesLUT = trainFunc(t_transforms, v_transforms, sentencesAudioInputMatrixTest, wordsTest, model, optimizer, lr_scheduler, epochIdx, validateOnly=True)
        testLoss.append(testLossEpoch)
        plotSentenceResults('Test', sentencesEstimationResultsTest, path2FigTest.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        fig = plt.subplots(figsize=(24, 10))
        firstEpochToPrint = 0
        plt.plot(epochVecTrain[firstEpochToPrint:epochIdx+1], trainLoss[firstEpochToPrint:epochIdx+1], label='train')
        plt.plot(epochVec, validateLoss, label='validate')
        plt.plot(epochVec, testLoss, label='test')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig('./trainLoss.png')
        # plt.show()
        plt.close()
        print(f'train loss: {trainLoss[epochIdx]}')

        print(f'train loss: {trainLoss[epochIdx]}; validation loss: {validateLoss[-1]}; test loss: {testLoss[-1]}')



