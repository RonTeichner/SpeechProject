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

path2sentencesAudioInputMatrixTrain = './sentencesAudioInputMatrixTrain.pt'
path2sentencesAudioInputMatrixValidate = './sentencesAudioInputMatrixValidate.pt'
path2sentencesAudioInputMatrixTest = './sentencesAudioInputMatrixTest.pt'

path2sentencesAudioInputMatrixLengthsTrain = './sentencesAudioInputMatrixLengthsTrain.pt'
path2sentencesAudioInputMatrixLengthsValidate = './sentencesAudioInputMatrixLengthsValidate.pt'
path2sentencesAudioInputMatrixLengthsTest = './sentencesAudioInputMatrixLengthsTest.pt'

path2MaxSentence = './maxSentence.pt'
path2AllLengths = './allLengths.pt'

enableWordOnlyClassificationAtEncoderOutput = False
enableTrain_wrt_groundTruth = False
enableSpectrogram = False

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

fs = 48000  # Hz
processingDuration = 25e-3  # sec
nTimeDomainSamplesInSingleFrame = int(processingDuration*fs)

sentencesEstimationResultsTrain = pickle.load(open(path2SentencesResultsTrain, "rb"))
sentencesEstimationResultsValidate = pickle.load(open(path2SentencesResultsValidate, "rb"))
sentencesEstimationResultsTest = pickle.load(open(path2SentencesResultsTest, "rb"))
'''
#findUniqueIndexes(path2SentencesMetadataTrain)
if os.path.isfile(path2MaxSentence):
    maxSentenceLengh = pickle.load(open(path2MaxSentence, "rb"))
else:
    sentencesDatasetsAudioTest = pickle.load(open(path2SentencesAudioTest, "rb"))
    sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
    sentencesDatasetsAudioTrain = pickle.load(open(path2SentencesAudioTrain, "rb"))
    maxSentenceLengh = 0
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioValidate):
        if len(sentence[1][1]) > maxSentenceLengh: maxSentenceLengh = len(sentence[1][1])
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioTest):
        if len(sentence[1][1]) > maxSentenceLengh: maxSentenceLengh = len(sentence[1][1])
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioTrain):
        if len(sentence[1][1]) > maxSentenceLengh: maxSentenceLengh = len(sentence[1][1])
    pickle.dump(maxSentenceLengh, open(path2MaxSentence, "wb"))

if os.path.isfile(path2AllLengths):
    allLengths = pickle.load(open(path2AllLengths, "rb"))
else:
    sentencesDatasetsAudioTest = pickle.load(open(path2SentencesAudioTest, "rb"))
    sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
    sentencesDatasetsAudioTrain = pickle.load(open(path2SentencesAudioTrain, "rb"))
    allLengths = np.zeros((len(sentencesDatasetsAudioTest) + len(sentencesDatasetsAudioValidate) + len(sentencesDatasetsAudioTrain)), dtype='int32')
    runningIdx = 0
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioValidate):
        allLengths[runningIdx] = len(sentence[1][1])
        runningIdx += 1
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioTest):
        allLengths[runningIdx] = len(sentence[1][1])
        runningIdx += 1
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudioTrain):
        allLengths[runningIdx] = len(sentence[1][1])
        runningIdx += 1
    pickle.dump(allLengths, open(path2AllLengths, "wb"))

n_bins = 100
n, bins, patches = plt.hist(allLengths, n_bins, density=True, histtype='step', cumulative=True, label='hist')
plt.xlabel('nSamplesInWord')
plt.grid(True)
plt.show()
'''
maxSentenceLengh = 48000 # this value is about 95% percentile of all datasets
# prepare the encoder input as a matrix by zero-padding the audio samples to have equal lengths:
if os.path.isfile(path2sentencesAudioInputMatrixValidate):
    sentencesAudioInputMatrixValidate = pickle.load(open(path2sentencesAudioInputMatrixValidate, "rb"))
    path2sentencesAudioInputMatrixLengthsValidate = pickle.load(open(path2sentencesAudioInputMatrixLengthsValidate, "rb"))
else:
    sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
    sentencesAudioInputMatrixValidate, sentencesAudioInputMatrixLengthsValidate = generateAudioMatrix(sentencesDatasetsAudioValidate, nTimeDomainSamplesInSingleFrame, enableSpectrogram, fs, maxSentenceLengh)
    pickle.dump(sentencesAudioInputMatrixValidate, open(path2sentencesAudioInputMatrixValidate, "wb"))
    pickle.dump(sentencesAudioInputMatrixLengthsValidate, open(path2sentencesAudioInputMatrixLengthsValidate, "wb"))

if os.path.isfile(path2sentencesAudioInputMatrixTest):
    sentencesAudioInputMatrixTest = pickle.load(open(path2sentencesAudioInputMatrixTest, "rb"))
    sentencesAudioInputMatrixLengthsTest = pickle.load(open(path2sentencesAudioInputMatrixLengthsTest, "rb"))
else:
    sentencesDatasetsAudioTest = pickle.load(open(path2SentencesAudioTest, "rb"))
    sentencesAudioInputMatrixTest, sentencesAudioInputMatrixLengthsTest = generateAudioMatrix(sentencesDatasetsAudioTest, nTimeDomainSamplesInSingleFrame, enableSpectrogram, fs, maxSentenceLengh)
    pickle.dump(sentencesAudioInputMatrixTest, open(path2sentencesAudioInputMatrixTest, "wb"))
    pickle.dump(sentencesAudioInputMatrixLengthsTest, open(path2sentencesAudioInputMatrixLengthsTest, "wb"))

if os.path.isfile(path2sentencesAudioInputMatrixTrain):
    sentencesAudioInputMatrixTrain = pickle.load(open(path2sentencesAudioInputMatrixTrain, "rb"))
    sentencesAudioInputMatrixLengthsTrain = pickle.load(open(path2sentencesAudioInputMatrixLengthsTrain, "rb"))
else:
    sentencesDatasetsAudioTrain = pickle.load(open(path2SentencesAudioTrain, "rb"))
    sentencesAudioInputMatrixTrain, sentencesAudioInputMatrixLengthsTrain = generateAudioMatrix(sentencesDatasetsAudioTrain, nTimeDomainSamplesInSingleFrame, enableSpectrogram, fs, maxSentenceLengh)
    '''
    listLen = len(sentencesDatasetsAudioTrain)
    batchLength = 10000
    nListBatches = int(np.ceil(listLen/batchLength))
    for listBatchIdx in range(nListBatches):
        if listBatchIdx == 0:
            sentencesAudioInputMatrixTrain = generateAudioMatrix(sentencesDatasetsAudioTrain[listBatchIdx*batchLength:(listBatchIdx+1)*batchLength], nTimeDomainSamplesInSingleFrame, enableSpectrogram, fs)
        else:
            sentencesAudioInputMatrixTrain = np.concatenate((sentencesAudioInputMatrixTrain, generateAudioMatrix(sentencesDatasetsAudioTrain[listBatchIdx*batchLength:min((listBatchIdx+1)*batchLength, listLen)], nTimeDomainSamplesInSingleFrame, enableSpectrogram, fs)), axis=1)
    '''
    pickle.dump(sentencesAudioInputMatrixTrain, open(path2sentencesAudioInputMatrixTrain, "wb"), protocol=4)
    pickle.dump(sentencesAudioInputMatrixLengthsTrain, open(path2sentencesAudioInputMatrixLengthsTrain, "wb"))


#model = VAE(measDim=nSamplesIn SingleLSTM_input, lstmHiddenSize=12, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=100, zDim=10).cuda()

# code to print an example wav signal:
'''
sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
wav = sentencesDatasetsAudioValidate[0][1][1]
wav01 = sentencesDatasetsAudioValidate[1][1][1]
tVec = np.arange(wav.shape[0])
tVec01 = np.arange(wav01.shape[0])
tVecTotal = np.arange(wav.shape[0]+wav01.shape[0])
wavTotal = np.concatenate((wav, wav01))
plt.plot(tVec, wav)
plt.xlabel('sec')
plt.title('Speech wav')
plt.savefig('./wavExample.png')

plt.plot(tVec01, wav01)
plt.xlabel('sec')
plt.title('Speech wav')
plt.savefig('./wav01Example.png')

plt.plot(tVecTotal, wavTotal)
plt.xlabel('sec')
plt.title('Speech wav')
plt.savefig('./wavTotalExample.png')
'''
if enableSpectrogram:
    nSamplesInSingleLSTM_input = sentencesAudioInputMatrixTrain.shape[-1]
else:
    nSamplesInSingleLSTM_input = nTimeDomainSamplesInSingleFrame

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
model = net_module.AudioCRNN(config=config, nDrawsFromSingleEncoderOutput=5, nDrawsFromSingleEncoderOutputEval=10, enableDiscrete=False).to('cuda')

trainable_params = filter(lambda p: p.requires_grad, model.parameters())

opt_name = config['optimizer']['type']
opt_args = config['optimizer']['args']
#opt_args['lr'] = 1e-3
optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

lr_name = config['lr_scheduler']['type']
lr_args = config['lr_scheduler']['args']
#lr_args['step_size'] = 1000
if lr_name == 'None':
    lr_scheduler = None
else:
    lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(optimizer, **lr_args)


#model = getattr(net_module, m_name)(classes, config=config)
#model = VAE(measDim=nSamplesInSingleLSTM_input, useLSTM=(nTimeIndexes > 1), lstmHiddenSize=40, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=1, zDim=512)#.cuda()
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0)

# model = VAE(measDim=nSamplesInSingleLSTM_input, lstmHiddenSize=40, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=1, zDim=40).cuda()
# with beta = 0 great results on train set after 2000 epochs. although the time tags are shuffled, bad results on validation set when it is not shuffled and also when it is shuffled
# so we should decrease the zDim and increase beta.
# beta = 0.1, zDim = 10:

# starting real mission - training on smoothing, canceling shuffle for start:
# beta = 1
# model = VAE(measDim=nSamplesInSingleLSTM_input, lstmHiddenSize=40, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=4, zDim=10).cuda()
# with beta = 1 the loss is fixed on above 7.
# try with beta = 0.1 - same result
# try with beta = 0.01 - after 2000 epochs it is same as smoothing on train set but bad on validation. I give it more time to see if it will generelize better. Maybe switch to SGD?

# same with SGD:
# after 2000 epochs the loss is above 7. Trying with lr=1e-4 (was 1e-3) - same after 1000. canceling momentum (was 0.9) and seeting beta=0 - loss above 7. it learns nothing witg SGD. why?

# Going back to Adam with lr - 1e-4 and beta = 0.05 - loss is stuck above 7 after 2000 epochs. trying with beta = 0.02. Nothing... try back with lr=1e-3.
# didn't work. goint back to beta = 0.01 - works but over fits. So maybe decrease zDim to 5 (was 10): still overfits. what about zDim=2 - nothing, loss above 7.
# zDim = 3 with beta = 0.01 - nothing. zDim = 3 with beta = 0: overfit. It seems to be hard to find the point where there is no overfit.


nSentencesForTrain = len(sentencesEstimationResultsTrain)

# sample from results dataset to obtain the 'deterministic' dataset:
sentencesEstimationResultsTrain_sampled = sampleFromSmoothing(sentencesEstimationResultsTrain, enableTrain_wrt_groundTruth)
sentencesEstimationPitchResultsTrain_sampled = sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, 3:4]#torch.tensor(sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, 3:4], dtype=torch.float16)#.cuda()
sentencesEstimationResultsTrain_sampled = sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, :3]#torch.tensor(sentencesEstimationResultsTrain_sampled[:nSentencesForTrain, :3], dtype=torch.uint8)#.cuda()
#sentencesAudioInputMatrixTrain = torch.tensor(sentencesAudioInputMatrixTrain[:, :nSentencesForTrain], dtype=torch.float32)#.cuda()
sentencesAudioInputMatrixTrain = sentencesAudioInputMatrixTrain[:nSentencesForTrain]

# variables for validation:
sentencesEstimationResultsValidate_sampled = sampleFromSmoothing(sentencesEstimationResultsValidate, enableTrain_wrt_groundTruth)
sentencesEstimationPitchResultsValidate_sampled = sentencesEstimationResultsValidate_sampled[:, 3:4]#torch.tensor(sentencesEstimationResultsValidate_sampled[:, 3:4], dtype=torch.float16)#.cuda()
sentencesEstimationResultsValidate_sampled = sentencesEstimationResultsValidate_sampled[:, :3]#torch.tensor(sentencesEstimationResultsValidate_sampled[:, :3], dtype=torch.uint8)#.cuda()
#sentencesAudioInputMatrixValidate = torch.tensor(sentencesAudioInputMatrixValidate, dtype=torch.float32)#.cuda()

# variables for test:
print('n time-tags in Spectrogram: %d' % len(sentencesAudioInputMatrixTest))
sentencesEstimationResultsTest_sampled = sampleFromSmoothing(sentencesEstimationResultsTest, enableTrain_wrt_groundTruth)
sentencesEstimationPitchResultsTest_sampled = sentencesEstimationResultsTest_sampled[:, 3:4]#torch.tensor(sentencesEstimationResultsTest_sampled[:, 3:4], dtype=torch.float16)#.cuda()
sentencesEstimationResultsTest_sampled = sentencesEstimationResultsTest_sampled[:, :3]#torch.tensor(sentencesEstimationResultsTest_sampled[:, :3], dtype=torch.uint8)#.cuda()
#sentencesAudioInputMatrixTest = torch.tensor(sentencesAudioInputMatrixTest, dtype=torch.float32)#.cuda()

# normalize pitch via the training set alone:
pitchScaler = StandardScaler()
pitchScaler.fit(sentencesEstimationPitchResultsTrain_sampled)
# transforming:
# n_bins = 20
# n, bins, patches = plt.hist(sentencesEstimationPitchResultsTrain_sampled, n_bins, density=True, histtype='step', cumulative=False, label='hist')
sentencesEstimationPitchResultsTrain_sampled = pitchScaler.transform(sentencesEstimationPitchResultsTrain_sampled)
# n, bins, patches = plt.hist(sentencesEstimationPitchResultsTrain_sampled, n_bins, density=True, histtype='step', cumulative=False, label='hist')
sentencesEstimationPitchResultsValidate_sampled = pitchScaler.transform(sentencesEstimationPitchResultsValidate_sampled)
sentencesEstimationPitchResultsTest_sampled = pitchScaler.transform(sentencesEstimationPitchResultsTest_sampled)


nEpochs = 1000+1
trainLoss = np.zeros(nEpochs)
epochVecTrain = np.arange(nEpochs)
validateLoss, testLoss, epochVec = list(), list(), list()
for epochIdx in range(nEpochs):
    trainLoss[epochIdx], _ = trainFunc(pitchScaler, t_transforms, v_transforms, beta, sentencesAudioInputMatrixTrain, sentencesEstimationResultsTrain_sampled, sentencesEstimationPitchResultsTrain_sampled, model, optimizer, lr_scheduler, epochIdx, validateOnly=False, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
    #trainLoss[epochIdx], _ = trainFunc(beta, sentencesAudioInputMatrixValidate, sentencesEstimationResultsValidate_sampled, sentencesEstimationPitchResultsValidate_sampled, model, optimizer, epochIdx, validateOnly=False, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
    if epochIdx % 10 == 0 and epochIdx > 0:
        epochVec.append(epochIdx)
        #try:

        _, probabilitiesLUT = trainFunc(pitchScaler, t_transforms, v_transforms, beta, sentencesAudioInputMatrixTrain[:1000], sentencesEstimationResultsTrain_sampled[:1000], sentencesEstimationPitchResultsTrain_sampled[:1000], model, optimizer, lr_scheduler, epochIdx, validateOnly=True, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
        plotSentenceResults('Train', sentencesEstimationResultsTrain[:1000], maleIdx, femaleIdx, path2FigValidate.split('.png')[0] + '_Trainepoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        validateLossEpoch, probabilitiesLUT = trainFunc(pitchScaler, t_transforms, v_transforms, beta, sentencesAudioInputMatrixValidate, sentencesEstimationResultsValidate_sampled, sentencesEstimationPitchResultsValidate_sampled, model, optimizer, lr_scheduler, epochIdx, validateOnly=True, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
        validateLoss.append(validateLossEpoch)
        plotSentenceResults('Validate', sentencesEstimationResultsValidate, maleIdx, femaleIdx, path2FigValidate.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        testLossEpoch, probabilitiesLUT = trainFunc(pitchScaler, t_transforms, v_transforms, beta, sentencesAudioInputMatrixTest, sentencesEstimationResultsTest_sampled, sentencesEstimationPitchResultsTest_sampled, model, optimizer, lr_scheduler, epochIdx, validateOnly=True, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
        testLoss.append(testLossEpoch)
        plotSentenceResults('Test', sentencesEstimationResultsTest, maleIdx, femaleIdx, path2FigTest.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)

        fig = plt.subplots(figsize=(24, 10))
        firstEpochToPrint = 10
        plt.plot(epochVecTrain[firstEpochToPrint:epochIdx+1], trainLoss[firstEpochToPrint:epochIdx+1], label='train')
        plt.plot(epochVec, validateLoss, label='validate')
        plt.plot(epochVec, testLoss, label='test')
        plt.legend()
        plt.xlabel('epochs')
        plt.savefig('./trainLoss.png')
        # plt.show()
        plt.close()
        print(f'train loss: {trainLoss[epochIdx]}')
        '''
        
        validateLoss, probabilitiesLUT = trainFunc(sentencesAudioInputMatrixTrain[:, :1000], sentencesEstimationResultsTrain_sampled[:1000], sentencesEstimationPitchResultsTrain_sampled[:1000], model, optimizer, epochIdx, validateOnly=True, enableSpectrogram=enableSpectrogram, enableSimpleClassification=enableWordOnlyClassificationAtEncoderOutput)
        if enableWordOnlyClassificationAtEncoderOutput:
            wordVecLUT, wordVecResults = wordVecFromProbabilitiesLUT(probabilitiesLUT), sentencesEstimationResultsTrain_sampled[:1000, -1].cpu().numpy()  # wordVecFromResults(sentencesEstimationResultsTrain[:1000])
            print('%% correct in epoch %d' % ((wordVecLUT.round() == wordVecResults.round()).sum()/len(wordVecLUT) * 100))
        plotSentenceResults(sentencesEstimationResultsTrain[:1000], maleIdx, femaleIdx, path2FigValidate.split('.png')[0] + '_epoch_%d' % epochIdx + '.png', sentencesEstimationResults_NN=probabilitiesLUT)
        '''
        print(f'train loss: {trainLoss[epochIdx]}; validation loss: {validateLoss[-1]}; test loss: {testLoss[-1]}')
        # print(f'train losses: {trainLoss[:epochIdx+1]}')

        #except:
        #    x=0

# Discussion:
'''
In 25msec @ fs = 48Khz there are 1200 samples
In 10msec @ fs = 48Khz there are 480 samples
So we want to insert the LSTM 1200 samples at each call and then jump 480 samples forward
But actually, we can trust the LSTM to deal with working with non-overlap samples.
Therefore we just want to reshape the data to have 1200 features instead of 1.
'''







