from audioMNIST_mata2Dict import *
from speakerfeatures import *
import matplotlib.pyplot as plt
from scipy.io import wavfile
import random
import os
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM
from sklearn.mixture import GaussianMixture


random.seed(0)

path2metadata = './AudioMNISTMetadata.pt'

path2metadataGenderDict = './AudioMNISTMetadataGender.pt'
path2GenderFeatures = './GenderFeatures.pt'
path2GenderAudio = './GenderAudio.pt'
path2GenderModels = './GenderModels.pt'

path2metadataSpeakerDict = './AudioMNISTMetadataSpeaker.pt'
path2SpeakerFeatures = './SpeakerFeatures.pt'
path2SpeakerAudio = './SpeakerAudio.pt'
path2SpeakerModels = './SpeakerModelsTmp.pt'

path2metadataWordDict = './AudioMNISTMetadataWord.pt'
path2WordFeatures = './WordFeatures.pt'
path2WordAudio = './WordAudio.pt'
path2WordModels = './WordModels.pt'

path2SentencesMetadata = './SentencesMetadata.pt'
path2SentencesAudio = './SentencesAudio.pt'
path2SentencesFeatures = './SentencesFeatures.pt'

path2SentencesPostEffectsFeatures = './SentencesPostEffectsFeatures.pt'
path2SentencesPostEffectsAudio = './SentencesPostEffectsAudio.pt'

path2SentencesResults = './SentencesResults.pt'
path2SentencesPostEffectsResults = './SentencesPostEffectsResults.pt'

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

metadata = pickle.load(open(path2metadata, "rb"))

wordDatasetsFeatures = pickle.load(open(path2WordFeatures, "rb"))
genderDatasetsFeatures = pickle.load(open(path2GenderFeatures, "rb"))
speakerDatasetsFeatures = pickle.load(open(path2SpeakerFeatures, "rb"))

sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
sentencesDatasetsFeatures = pickle.load(open(path2SentencesFeatures, "rb"))

caseStudy = 'speaker' #'gender'# 'speaker' 'word'

maxDiff = -np.inf
while True:
    nFeatures2Choose = np.arange(start=1, stop=5)
    np.random.shuffle(nFeatures2Choose)
    nFeatures2Choose = nFeatures2Choose[0]

    #chosenFeatures = np.arange(start=2, stop=39)
    chosenFeatures = np.arange(start=1, stop=39)
    np.random.shuffle(chosenFeatures)
    #chosenFeatures = np.concatenate([chosenFeatures[:nFeatures2Choose], [1]])
    chosenFeatures = chosenFeatures[:nFeatures2Choose]

    if caseStudy == 'word':
        categoryClassificationTrain(wordDatasetsFeatures, path2WordModels, 'words', True, chosenFeatures)
    elif caseStudy == 'gender':
        categoryClassificationTrain(genderDatasetsFeatures, path2GenderModels, 'speaker', True, chosenFeatures)
    elif caseStudy == 'speaker':
        categoryClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels, 'speaker', True, chosenFeatures)

    sentencesEstimationResults = createSentencesEstimationResults(sentencesDatasetsFeatures[:100], None, metadata, path2SentencesResults, path2WordModels, path2SpeakerModels, path2GenderModels, transitionMat, priorStates, trainOnLessFeatures=True, enableMahalanobisCala=True, chosenFeatures=chosenFeatures)

    meanFiltering, meanSmoothing = meanFilteringSmoothingCalc(sentencesEstimationResults, maleIdx, femaleIdx, classCategories=caseStudy)

    currentDiff = 100*(meanSmoothing-meanFiltering)
    if currentDiff > maxDiff:
        maxDiff = currentDiff.copy()
        bestFeatures = chosenFeatures.copy()
    print(f'currentFeatures: {chosenFeatures}; currentDiff: {currentDiff}; bestFeatures: {bestFeatures}; maxDiff: {maxDiff}')


