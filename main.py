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

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

enableGenderTrain = True
enableSpeakerTrain = True
enableWordDetection = True
enableSentenceDetection = True

enablePureSentenceTest = True
enablePureSentencePlots = True

#enableSentencePostEffectFeaturesCreation = False
#enableEffectSentencePlots = False

chosenFeatures = [1, 13, 21] # best word features

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

numberOfTrainWords(metadata)

# code for ploting MFCC:
'''
genderDatasetsFeatures = pickle.load(open(path2GenderFeatures, "rb"))
plt.imshow(genderDatasetsFeatures['train'][0][0][0:100,:].transpose(), aspect='auto')
plt.xlabel('sec')
plt.ylabel('MFCC')
plt.title('Speech MFCC')
plt.savefig('MFCC_example.png')
'''
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
        genderModels = categoryClassificationTrain(genderDatasetsFeatures, path2GenderModels, type='speaker', trainOnLessFeatures=True, chosenFeatures=chosenFeatures)

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
        speakerModels = categoryClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels, type='speaker', trainOnLessFeatures=True, chosenFeatures=chosenFeatures)

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
        wordModels = categoryClassificationTrain(wordDatasetsFeatures, path2WordModels, type='words', trainOnLessFeatures=True, chosenFeatures=chosenFeatures)

if enableSentenceDetection:
    sentencesEstimationResultsTrain = createSentencesDataset(metadata, path2SentencesResultsTrain, path2SentencesMetadataTrain, path2SentencesFeaturesTrain, path2SentencesAudioTrain, path2SentencesPitchTrain, path2WordModels, path2SpeakerModels, path2GenderModels, chosenFeatures, nSentences=17900, whichSet='train')
    sentencesEstimationResultsValidate = createSentencesDataset(metadata, path2SentencesResultsValidate, path2SentencesMetadataValidate, path2SentencesFeaturesValidate, path2SentencesAudioValidate, path2SentencesPitchValidate, path2WordModels, path2SpeakerModels, path2GenderModels, chosenFeatures, nSentences=2000, whichSet='validate')
    createSentencesDataset(metadata, path2SentencesResultsTest, path2SentencesMetadataTest, path2SentencesFeaturesTest, path2SentencesAudioTest, path2SentencesPitchTest, path2WordModels, path2SpeakerModels, path2GenderModels, chosenFeatures, nSentences=2000, whichSet='test')

if enablePureSentencePlots:
    plotSentenceResults(pickle.load(open(path2SentencesResultsTrain, "rb")), maleIdx, femaleIdx, path2FigTrain)
    plotSentenceResults(pickle.load(open(path2SentencesResultsValidate, "rb")), maleIdx, femaleIdx, path2FigValidate)
    plotSentenceResults(pickle.load(open(path2SentencesResultsTest, "rb")), maleIdx, femaleIdx, path2FigTest)
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

# plotPitchHistogramPerSentence(pickle.load(open(path2SentencesPitch, "rb")), pickle.load(open(path2SentencesResults, "rb")))
