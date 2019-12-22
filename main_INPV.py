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

sentencesDatasetsAudioTrain = pickle.load(open(path2SentencesAudioTrain, "rb"))
sentencesEstimationResultsTrain = pickle.load(open(path2SentencesResultsTrain, "rb"))

sentencesDatasetsAudioValidate = pickle.load(open(path2SentencesAudioValidate, "rb"))
sentencesEstimationResultsValidate = pickle.load(open(path2SentencesResultsValidate, "rb"))


class VAE(nn.Module):
    def __init__(self, measDim, lstmHiddenSize, lstmNumLayers, nDrawsFromSingleEncoderOutput, zDim):
        super(VAE, self).__init__()

        self.nDrawsFromSingleEncoderOutput = nDrawsFromSingleEncoderOutput

        self.zDim = zDim
        self.decoderInnerWidth = 5
        self.genderMultivariate = 2
        self.speakerMultivariate = 60
        self.wordMultivariate = 10
        self.nContinuesParameters = 1 # pitch

        # encoder:
        self.kalmanLSTM = nn.LSTM(input_size=measDim, hidden_size=lstmHiddenSize, num_layers=lstmNumLayers)
        self.fc21 = nn.Linear(lstmHiddenSize, self.zDim)
        self.fc22 = nn.Linear(lstmHiddenSize, self.zDim)

        # decoder:
        self.fc3 = nn.Linear(self.zDim, self.decoderInnerWidth)
        # self.fc4 = nn.Linear(self.decoderInnerWidth, self.decoderInnerWidth)
        self.fc5 = nn.Linear(self.decoderInnerWidth, self.genderMultivariate + self.speakerMultivariate + self.wordMultivariate + 2*self.nContinuesParameters)

        # general:
        self.logSoftMax = nn.LogSoftmax(dim=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()


    def encode(self, y):
        output, (hn, cn) = self.kalmanLSTM(y)
        return self.fc21(hn[0]), self.fc22(hn[0])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.LeakyReLU(self.fc3(z))
        #h4 = self.tanh(self.fc4(h3))
        #return self.logSoftMax(self.fc5(h3)) # h3 here performed good without p(z) constrain
        #return torch.sigmoid(self.fc5(h3))  # h3 here performed good without p(z) constrain
        h5 = self.fc5(h3)
        return h5 # h3 here performed good without p(z) constrain

    def forward(self, y):
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu.repeat(self.nDrawsFromSingleEncoderOutput, 1), logvar.repeat(self.nDrawsFromSingleEncoderOutput, 1))
        decoderOut = self.decode(z)
        genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar = decoderOut[:, :self.genderMultivariate], decoderOut[:, self.genderMultivariate:self.genderMultivariate+self.speakerMultivariate], decoderOut[:, self.genderMultivariate+self.speakerMultivariate : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate], decoderOut[:, self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+1], decoderOut[:, self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+1 : self.genderMultivariate+self.speakerMultivariate+self.wordMultivariate+2]
        return genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar #, mu, logvar, z


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def loss_function(genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, sampledGender, sampledSpeaker, sampledWord, sampledPitch, nDecoders):

    sampledWord, sampledGender, sampledSpeaker, sampledPitch = sampledWord.unsqueeze(1).expand(-1, nDecoders).reshape(-1), sampledGender.unsqueeze(1).expand(-1, nDecoders).reshape(-1), sampledSpeaker.unsqueeze(1).expand(-1, nDecoders).reshape(-1), sampledPitch.unsqueeze(1).expand(-1, nDecoders).reshape(-1)

    genderNLL = F.cross_entropy(genderProbs, sampledGender, reduction='none')
    speakerNLL = F.cross_entropy(speakerProbs, sampledSpeaker, reduction='none')
    wordNLL = F.cross_entropy(wordProbs, sampledWord, reduction='none')

    pitchNLL = torch.zeros(pitchMean.shape)
    for pitchIdx in range(pitchMean.shape[0]):
        pitchNLL[pitchIdx] = torch.distributions.normal.Normal(pitchMean[pitchIdx], pitchLogVar[pitchIdx].mul(0.5).exp_()).log_prob(sampledPitch[pitchIdx])

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def trainFunc(sentencesAudioInputMatrixTrain, sentencesEstimationResultsTrain_sampled, model, optimizer):
    model.train()
    train_loss = 0

    batchSize = 200
    nSentences = sentencesAudioInputMatrixTrain.shape[0]
    nBatches = int(nSentences/batchSize)
    nDecoders = model.nDrawsFromSingleEncoderOutput

    inputSentenceIndexes = torch.randperm(nSentences)
    sentencesAudioInputMatrixTrain, sentencesEstimationResultsTrain_sampled = sentencesAudioInputMatrixTrain[inputSentenceIndexes], sentencesEstimationResultsTrain_sampled[inputSentenceIndexes]

    for batchIdx in range(nBatches):
        data = sentencesAudioInputMatrixTrain[batchIdx * batchSize:(batchIdx + 1) * batchSize].transpose(1,0).unsqueeze_(2)
        labels = sentencesEstimationResultsTrain_sampled[batchIdx * batchSize:(batchIdx + 1) * batchSize]
        sampledGender, sampledSpeaker, sampledWord, sampledPitch = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]

        optimizer.zero_grad()
        genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar = model(data)
        loss = loss_function(genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, sampledGender, sampledSpeaker, sampledWord, sampledPitch, model.nDrawsFromSingleEncoderOutput)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        x=3
    return reconBatch




# sample from results dataset to obtain the 'deterministic' dataset:
sentencesEstimationResultsTrain_sampled = sampleFromSmoothing(sentencesEstimationResultsTrain)

# prepare the encoder input as a matrix by zero-padding the audio samples to have equal lengths:
sentencesAudioInputMatrixTrain = generateAudioMatrix(sentencesDatasetsAudioTrain)

model = VAE(measDim=1, lstmHiddenSize=12, lstmNumLayers=1, nDrawsFromSingleEncoderOutput=100, zDim=10).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
sentencesEstimationResultsTrain_sampled = torch.tensor(sentencesEstimationResultsTrain_sampled, dtype=torch.long).cuda()
sentencesAudioInputMatrixTrain = torch.tensor(sentencesAudioInputMatrixTrain, dtype=torch.float32).cuda()

nEpochs = 2
for epochIdx in range(nEpochs):
    trainFunc(sentencesAudioInputMatrixTrain, sentencesEstimationResultsTrain_sampled, model, optimizer)
x = 3





