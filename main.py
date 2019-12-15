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

path2SentencesMetadata = './SentencesMetadata.pt'
path2SentencesAudio = './SentencesAudio.pt'
path2SentencesFeatures = './SentencesFeatures.pt'
path2SentencesPitch = './SentencesPitch.pt'

path2SentencesPostEffectsFeatures = './SentencesPostEffectsFeatures.pt'
path2SentencesPostEffectsAudio = './SentencesPostEffectsAudio.pt'

path2SentencesResults = './SentencesResults.pt'
path2SentencesPostEffectsResults = './SentencesPostEffectsResults.pt'

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

enableGenderTrain = False
enableSpeakerTrain = False
enableWordDetection = True
enableSentenceDetection = True

enablePureSentenceTest = False
enablePureSentencePlots = False

#enableSentencePostEffectFeaturesCreation = False
#enableEffectSentencePlots = False

# create/load metadata:
if os.path.isfile(path2metadata):
    metadata = pickle.load(open(path2metadata, "rb"))
else:
    metadata = AudioMNISTMetaData()
    trainPortion, validatePortion, testPortion = 0.6, 0.1, 0.3
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
        genderDatasetsFeatures = createcategoryWavs_Features(metadata, fs, path2GenderAudio, path2GenderFeatures, 'genders')

    # train gender detection:
    if os.path.isfile(path2GenderModels):
        genderModels = pickle.load(open(path2GenderModels, "rb"))
    else:
        categoryClassificationTrain(genderDatasetsFeatures, path2GenderModels)

if enableSpeakerTrain:
    # create\load speakers features:
    if os.path.isfile(path2SpeakerFeatures):
        speakerDatasetsFeatures = pickle.load(open(path2SpeakerFeatures, "rb"))
        # speakerDatasetsAudio = pickle.load(open(path2SpeakerAudio, "rb"))
    else:
        speakerDatasetsFeatures = createcategoryWavs_Features(metadata, fs, path2SpeakerAudio, path2SpeakerFeatures, 'speakers')

    # train speakers detection:
    if os.path.isfile(path2SpeakerModels):
        speakerModels = pickle.load(open(path2SpeakerModels, "rb"))
    else:
        categoryClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels)

if enableWordDetection:
    # create\load speakers features:
    if os.path.isfile(path2WordFeatures):
        wordDatasetsFeatures = pickle.load(open(path2WordFeatures, "rb"))
        # speakerDatasetsAudio = pickle.load(open(path2SpeakerAudio, "rb"))
        # sd.play(speakerDatasetsAudio['test'][0][0],fs)
        # sd.stop()
    else:
        wordDatasetsFeatures = createcategoryWavs_Features(metadata, fs, path2WordAudio, path2WordFeatures, 'words')

    # train speakers detection:
    if os.path.isfile(path2WordModels):
        wordModels = pickle.load(open(path2WordModels, "rb"))
    else:
        wordModels = categoryClassificationTrain(wordDatasetsFeatures, path2WordModels, 'words', trainOnLessFeatures=False)

if enableSentenceDetection:
    if os.path.isfile(path2SentencesResults):
        sentencesEstimationResults = pickle.load(open(path2SentencesResults, "rb"))
        sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
    else:
        # create sentences dataset:
        if os.path.isfile(path2SentencesMetadata):
            sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
            # sentencesDatasetsAudio = pickle.load(open(path2SentencesAudio, "rb"))
            # sd.play(sentencesDatasetsAudio[0][1][1],fs)
            # sd.stop()
        else:
            sentencesMetadata, priorStates, transitionMat = createSentencesMetadata(metadata, path2SentencesMetadata)

        # create\load sentences features:
        if os.path.isfile(path2SentencesFeatures):
            sentencesDatasetsFeatures = pickle.load(open(path2SentencesFeatures, "rb"))
        else:
            sentencesDatasetsFeatures = createSentenceWavs_Features(sentencesMetadata, path2SentencesAudio, path2SentencesFeatures)

        # create\load sentences pitch:
        if os.path.isfile(path2SentencesPitch):
            sentencesDatasetsPitch = pickle.load(open(path2SentencesPitch, "rb"))
            # plotPitchHistogramPerSentence(sentencesDatasetsPitch)
        else:
            sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
            sentencesDatasetsPitch = createSentenceWavs_Features(sentencesMetadata, None, path2SentencesPitch, includeEffects=False, createPitch=True)
            # plotPitchHistogramPerSentence(sentencesDatasetsPitch)
            # The conclusion from plotting the pitch histograms per sentence is to model the sentence speach by a mixture of two Gaussians

        # create\load sentences estimation results:
        if os.path.isfile(path2SentencesResults):
            sentencesEstimationResults = pickle.load(open(path2SentencesResults, "rb"))
        else:
            sentencesEstimationResults = createSentencesEstimationResults(sentencesDatasetsFeatures, sentencesDatasetsPitch, metadata, path2SentencesResults, path2WordModels, path2SpeakerModels, path2GenderModels, transitionMat, priorStates, trainOnLessFeatures=False, enableMahalanobisCala=False)

if enablePureSentencePlots:
    sentencesEstimationResults = pickle.load(open(path2SentencesResults, "rb"))
    plotSentenceResults(sentencesEstimationResults, maleIdx, femaleIdx)
'''
if enableSentencePostEffectFeaturesCreation:

    # create\load sentences features:
    if os.path.isfile(path2SentencesPostEffectsFeatures):
        sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
        sentencesDatasetsPostEffectsFeatures = pickle.load(open(path2SentencesPostEffectsFeatures, "rb"))
    else:
        sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
        sentencesDatasetsPostEffectsFeatures = createSentenceWavs_Features(sentencesMetadata, path2SentencesPostEffectsAudio, path2SentencesPostEffectsFeatures, includeEffects=True)

    # create\load sentences estimation results:
    if os.path.isfile(path2SentencesPostEffectsResults):
        sentencesPostEffectsEstimationResults = pickle.load(open(path2SentencesPostEffectsResults, "rb"))
    else:
        metadata = pickle.load(open(path2metadata, "rb"))
        sentencesPostEffectsEstimationResults = createSentencesEstimationResults(sentencesDatasetsPostEffectsFeatures, metadata, path2SentencesPostEffectsResults, path2WordModels, path2SpeakerModels, path2GenderModels, transitionMat, priorStates, trainOnLessFeatures=True, enableMahalanobisCala=True)

if enableEffectSentencePlots:
    sentencesPostEffectsEstimationResults = pickle.load(open(path2SentencesPostEffectsResults, "rb"))
    plotSentenceResults(sentencesPostEffectsEstimationResults, maleIdx, femaleIdx)
'''
# wordFeatureHistograms(path2WordFeatures)
