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

random.seed(0)

path2metadata = './AudioMNISTMetadata.pt'

path2metadataGenderDict = './AudioMNISTMetadataGender.pt'
path2GenderFeatures = './GenderFeatures.pt'
path2GenderAudio = './GenderAudio.pt'
path2GenderModels = './GenderModels.pt'

path2metadataSpeakerDict = './AudioMNISTMetadataSpeaker.pt'
path2SpeakerFeatures = './SpeakerFeatures.pt'
path2SpeakerAudio = './SpeakerAudio.pt'
path2SpeakerModels = './SpeakerModels.pt'

path2metadataWordDict = './AudioMNISTMetadataWord.pt'
path2WordFeatures = './WordFeatures.pt'
path2WordAudio = './WordAudio.pt'
path2WordModels = './WordModels.pt'

path2SentencesMetadataTrain = './SentencesMetadataTrain.pt'
path2SentencesAudioTrain = './SentencesAudioTrain.pt'
path2SentencesFeaturesTrain = './SentencesFeaturesTrain.pt'
path2SentencesPitchTrain = './SentencesPitchTrain.pt'
path2SentencesResultsTrain = './SentencesResultsTrain.pt'
path2FigTrain = './SentencesFigTrain.png'

path2SentencesMetadataValidate = './SentencesMetadataValidate.pt'
path2SentencesAudioValidate = './SentencesAudioValidate.pt'
path2SentencesFeaturesValidate = './SentencesFeaturesValidate.pt'
path2SentencesPitchValidate = './SentencesPitchValidate.pt'
path2SentencesResultsValidate = './SentencesResultsValidate.pt'
path2FigValidate = './SentencesFigValidate.png'

path2SentencesMetadataTest = './SentencesMetadataTest.pt'
path2SentencesAudioTest = './SentencesAudioTest.pt'
path2SentencesFeaturesTest = './SentencesFeaturesTest.pt'
path2SentencesPitchTest = './SentencesPitchTest.pt'
path2SentencesResultsTest = './SentencesResultsTest.pt'
path2FigTest = './SentencesFigTest.png'

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

sentencesDatasetsAudioTrain = pickle.load(open(path2SentencesAudioTrain, "rb"))
sentencesEstimationResultsTrain = pickle.load(open(path2SentencesResultsTrain, "rb"))

sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
sentencesEstimationResultsValidate = pickle.load(open(path2SentencesResultsValidate, "rb"))

# sample from results dataset to obtain the 'deterministic' dataset:
sentencesEstimationResultsTrain_sampled = sampleFromSmoothing(sentencesEstimationResultsTrain)

# prepare the encoder input as a matrix by zero-padding the audio samples to have equal lengths:
sentencesAudioInputMatrixTrain = generateAudioMatrix(sentencesDatasetsAudioTrain)

fs = 48000 # Hz
processingDuration = 25e-3 # sec
nSamplesInSingleLSTM_input = int(processingDuration*fs)

#model = VAE(measDim=nSamplesInSingleLSTM_input, lstmHiddenSize=12, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=100, zDim=10).cuda()
model = VAE(measDim=nSamplesInSingleLSTM_input, lstmHiddenSize=40, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=1, zDim=10).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

nSentencesForTrain = sentencesEstimationResultsTrain_sampled.shape[0]  # 1000

sentencesEstimationPitchResultsTrain_sampled = torch.tensor(sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, 3:4], dtype=torch.float16).cuda()
sentencesEstimationResultsTrain_sampled = torch.tensor(sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, :3], dtype=torch.uint8).cuda()
sentencesAudioInputMatrixTrain = torch.tensor(sentencesAudioInputMatrixTrain[:nSentencesForTrain], dtype=torch.int16).cuda()

# variables for validation:
sentencesEstimationResultsValidate_sampled = sampleFromSmoothing(sentencesEstimationResultsValidate)
sentencesAudioInputMatrixValidate = torch.tensor(generateAudioMatrix(sentencesDatasetsAudioValidate), dtype=torch.int16).cuda()
sentencesEstimationPitchResultsValidate_sampled = torch.tensor(sentencesEstimationResultsValidate_sampled[:, 3:4], dtype=torch.float16).cuda()
sentencesEstimationResultsValidate_sampled = torch.tensor(sentencesEstimationResultsValidate_sampled[:, :3], dtype=torch.uint8).cuda()


nEpochs = 10000
trainLoss = np.zeros(nEpochs)
for epochIdx in range(nEpochs):
    trainLoss[epochIdx], _ = trainFunc(sentencesAudioInputMatrixTrain, sentencesEstimationResultsTrain_sampled, sentencesEstimationPitchResultsTrain_sampled, model, optimizer, epochIdx, validateOnly=False)
    if epochIdx % 100 == 0:
        #validateLoss, probabilitiesLUT = trainFunc(sentencesAudioInputMatrixValidate, sentencesEstimationResultsValidate_sampled, sentencesEstimationPitchResultsValidate_sampled, model, optimizer, epochIdx, validateOnly=True)
        validateLoss, probabilitiesLUT = trainFunc(sentencesAudioInputMatrixTrain[:1000], sentencesEstimationResultsTrain_sampled[:1000], sentencesEstimationPitchResultsTrain_sampled[:1000], model, optimizer, epochIdx, validateOnly=True)
        # print(f'train losses: {trainLoss[:epochIdx+1]}')
        fig = plt.subplots(figsize=(24, 10))
        plt.plot(trainLoss[:epochIdx + 1])
        plt.xlabel('epochs')
        plt.savefig('./trainLoss.png')
        plt.show()
        print(f'train loss: {trainLoss[epochIdx]}; validation loss: {validateLoss}')
        plotSentenceResults(sentencesEstimationResultsValidate, maleIdx, femaleIdx, path2FigValidate.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

# Discussion:
'''
In 25msec @ fs = 48Khz there are 1200 samples
In 10msec @ fs = 48Khz there are 480 samples
So we want to insert the LSTM 1200 samples at each call and then jump 480 samples forward
But actually, we can trust the LSTM to deal with working with non-overlap samples.
Therefore we just want to reshape the data to have 1200 features instead of 1.
'''







