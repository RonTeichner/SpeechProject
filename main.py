from audioMNIST_mata2Dict import *
from speakerfeatures import *
import matplotlib.pyplot as plt
import random
import os
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM
from sklearn.mixture import GaussianMixture
import sounddevice as sd

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

path2SentencesMetadata = './SentencesMetadata.pt'
path2SentencesAudio = './SentencesAudio.pt'
path2SentencesFeatures = './SentencesFeatures.pt'

path2SentencesResults = './SentencesResults'

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

enableGenderTrain = False
enableSpeakerTrain = False
enableWordDetection = False
enableSentenceDetection = True

# create/load metadata:
if os.path.isfile(path2metadata):
    metadata = pickle.load(open(path2metadata, "rb"))
else:
    metadata = AudioMNISTMetaData()
    trainPortion, validatePortion, testPortion = 0.1, 0.1, 0.2
    metadata.label_train_sets(trainPortion, validatePortion, testPortion, genderEqual=False)
    pickle.dump(metadata, open(path2metadata, "wb"))

fs = metadata.fs  # [hz]
print('nMales, nFemales = %d, %d' % metadata.get_number_of_males_females())

if enableGenderTrain:
    # create\load gender features:
    if os.path.isfile(path2GenderFeatures):
        genderDatasetsFeatures = pickle.load(open(path2GenderFeatures, "rb"))
        #genderDatasetsAudio = pickle.load(open(path2GenderAudio, "rb"))
    else:
        genderDatasetsFeatures = createSpeakerWavs_Features(metadata, fs, path2GenderAudio, path2GenderFeatures, 'genders')

    # train gender detection:
    if os.path.isfile(path2GenderModels):
        genderModels = pickle.load(open(path2GenderModels, "rb"))
    else:
        speakerClassificationTrain(genderDatasetsFeatures, path2GenderModels)

if enableSpeakerTrain:
    # create\load speakers features:
    if os.path.isfile(path2SpeakerFeatures):
        speakerDatasetsFeatures = pickle.load(open(path2SpeakerFeatures, "rb"))
        # speakerDatasetsAudio = pickle.load(open(path2SpeakerAudio, "rb"))
    else:
        speakerDatasetsFeatures = createSpeakerWavs_Features(metadata, fs, path2SpeakerAudio, path2SpeakerFeatures, 'speakers')

    # train speakers detection:
    if os.path.isfile(path2SpeakerModels):
        speakerModels = pickle.load(open(path2SpeakerModels, "rb"))
    else:
        speakerClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels)

if enableWordDetection:
    # create\load speakers features:
    if os.path.isfile(path2WordFeatures):
        wordDatasetsFeatures = pickle.load(open(path2WordFeatures, "rb"))
        # speakerDatasetsAudio = pickle.load(open(path2SpeakerAudio, "rb"))
        # sd.play(speakerDatasetsAudio['test'][0][0],fs)
        # sd.stop()
    else:
        wordDatasetsFeatures = createSpeakerWavs_Features(metadata, fs, path2WordAudio, path2WordFeatures, 'words')

    # train speakers detection:
    if os.path.isfile(path2WordModels):
        wordModels = pickle.load(open(path2WordModels, "rb"))
    else:
        speakerClassificationTrain(wordDatasetsFeatures, path2WordModels, 'words')

if enableSentenceDetection:
    if os.path.isfile(path2SentencesResults):
        sentencesEstimationResults = pickle.load(open(path2SentencesResults, "rb"))
    else:
        # create sentences dataset:
        if os.path.isfile(path2SentencesMetadata):
            sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
            # sentencesDatasetsAudio = pickle.load(open(path2SentencesAudio, "rb"))
            # sd.play(wordDatasetsAudio['train'][0][0],fs)
            # sd.stop()
        else:
            sentencesMetadata, priorStates, transitionMat = createSentencesMetadata(metadata, path2SentencesMetadata)

        # create\load sentences features:
        if os.path.isfile(path2SentencesFeatures):
            sentencesDatasetsFeatures = pickle.load(open(path2SentencesFeatures, "rb"))
        else:
            sentencesDatasetsFeatures = createSentenceWavs_Features(sentencesMetadata, path2SentencesAudio, path2SentencesFeatures)

        # load models:
        wordModels = pickle.load(open(path2WordModels, "rb"))
        speakerModels = pickle.load(open(path2SpeakerModels, "rb"))
        genderModels = pickle.load(open(path2GenderModels, "rb"))

        nGenderModels = len(genderModels)
        genderSentenceModel = GMMHMM(n_components=nGenderModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nGenderModels)) # fit creates all internal variables
        genderSentenceModel.transmat_, genderSentenceModel.startprob_ = np.eye(nGenderModels), np.ones(nGenderModels)/nGenderModels

        nSpeakersModels = len(speakerModels)
        speakerSentenceModel = GMMHMM(n_components=nSpeakersModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nSpeakersModels))  # fit creates all internal variables
        speakerSentenceModel.transmat_, speakerSentenceModel.startprob_ = np.eye(nSpeakersModels), np.ones(nSpeakersModels)/nSpeakersModels

        nWordModels = len(wordModels)
        wordSentenceModel = GMMHMM(n_components=nWordModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nWordModels))  # fit creates all internal variables
        wordSentenceModel.transmat_, wordSentenceModel.startprob_ = transitionMat, priorStates



        sentencesEstimationResults = list()
        for sentenceIdx, sentence in enumerate(sentencesDatasetsFeatures):
            if sentenceIdx % 100 == 0:
                print('starting sentence %d estimation out of %d' % (sentenceIdx, len(sentencesDatasetsFeatures)))
            sentenceDict = dict()
            nWords = len(sentence)
            sentenceDict['groundTruth'] = dict()
            sentenceDict['groundTruth']['SpeakerNo'] = sentence[0]
            sentenceDict['groundTruth']['SpeakerGender'] = metadata.metaDataDict[sentence[0]]['gender']
            sentenceDict['groundTruth']['Digits'] = [sentence[i][0] for i in range(1, nWords)]

            sentenceDict['results'] = dict()

            sentenceDict['results']['gender'] = dict()
            sentenceDict['results']['gender']['filtering'], sentenceDict['results']['gender']['smoothing'] = computeFilteringSmoothing(genderModels, sentence, genderSentenceModel)

            sentenceDict['results']['speaker'] = dict()
            sentenceDict['results']['speaker']['filtering'], sentenceDict['results']['speaker']['smoothing'] = computeFilteringSmoothing(speakerModels, sentence, speakerSentenceModel)

            sentenceDict['results']['word'] = dict()
            sentenceDict['results']['word']['filtering'], sentenceDict['results']['word']['smoothing'] = computeFilteringSmoothing(wordModels, sentence, wordSentenceModel)

            sentencesEstimationResults.append(sentenceDict)
        pickle.dump(sentencesEstimationResults, open(path2SentencesResults, "wb"))

nSentences = len(sentencesEstimationResults)
classCategories = ['word', 'gender', 'speaker']
collectedFirstWordSentenceResults = dict()
for estimationClass in classCategories:
    collectedFirstWordSentenceResults[estimationClass] = dict()
    firstDigit_filtering, firstDigit_smoothing = np.zeros(nSentences), np.zeros(nSentences)
    for sentenceIdx in range(nSentences):
        sentenceResult = sentencesEstimationResults[sentenceIdx]
        if estimationClass == 'word':
            trueFirstDigit = sentenceResult['groundTruth']['Digits'][0]
            firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueFirstDigit]
            firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueFirstDigit]
        elif estimationClass == 'gender':
            trueGender = sentenceResult['groundTruth']['SpeakerGender']
            if trueGender == 'male':
                trueGenderIdx = maleIdx
            else:
                trueGenderIdx = femaleIdx
            firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueGenderIdx]
            firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueGenderIdx]
        elif estimationClass == 'speaker':
            trueSpeakerNo = int(sentenceResult['groundTruth']['SpeakerNo']) - 1
            firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueSpeakerNo]
            firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueSpeakerNo]
    collectedFirstWordSentenceResults[estimationClass]['filtering'] = firstDigit_filtering
    collectedFirstWordSentenceResults[estimationClass]['smoothing'] = firstDigit_smoothing

fig = plt.subplots(figsize=(16, 4))
for plotIdx, estimationClass in enumerate(classCategories):
    plt.subplot(1, len(classCategories), plotIdx+1)
    n_bins = 100
    n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['filtering'], n_bins, density=True, histtype='step', cumulative=True, label='Filtering')
    n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['smoothing'], n_bins, density=True, histtype='step', cumulative=True, label='Smoothing')
    plt.grid(True)
    plt.legend(loc='right')
    plt.title('likelihood CDF: ' + estimationClass)
    plt.xlabel('likelihood')
plt.show()



