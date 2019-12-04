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

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

enableGenderTrain = False
enableSpeakerTrain = False
enableWordDetection = True

if enableGenderTrain:
    # create/load metadata:
    if os.path.isfile(path2metadataGenderDict):
        metadataGender = pickle.load(open(path2metadataGenderDict, "rb"))
    else:
        metadataGender = AudioMNISTMetaData()
        trainPortion, validatePortion, testPortion = 0.1, 0.1, 0.1
        metadataGender.label_train_sets(trainPortion, validatePortion, testPortion, genderEqual=True)
        pickle.dump(metadataGender, open(path2metadataGenderDict, "wb"))

    fs = metadataGender.fs  # [hz]
    print('nMales, nFemales = %d, %d' % metadataGender.get_number_of_males_females())

    # create\load gender features:
    if os.path.isfile(path2GenderFeatures):
        genderDatasetsFeatures = pickle.load(open(path2GenderFeatures, "rb"))
        #genderDatasetsAudio = pickle.load(open(path2GenderAudio, "rb"))
    else:
        genderDatasetsFeatures = createSpeakerWavs_Features(metadataGender, fs, path2GenderAudio, path2GenderFeatures, 'genders')

    # train gender detection:
    if os.path.isfile(path2GenderModels):
        genderModels = pickle.load(open(path2GenderModels, "rb"))
    else:
        speakerClassificationTrain(genderDatasetsFeatures, path2GenderModels)

if enableSpeakerTrain:
    # create/load metadata:
    if os.path.isfile(path2metadataSpeakerDict):
        metadataSpeaker = pickle.load(open(path2metadataSpeakerDict, "rb"))
    else:
        metadataSpeaker = AudioMNISTMetaData()
        trainPortion, validatePortion, testPortion = 0.1, 0.1, 0.1
        metadataSpeaker.label_train_sets(trainPortion, validatePortion, testPortion, genderEqual=False)
        pickle.dump(metadataSpeaker, open(path2metadataSpeakerDict, "wb"))

    fs = metadataSpeaker.fs  # [hz]
    print('nMales, nFemales = %d, %d' % metadataSpeaker.get_number_of_males_females())

    # create\load speakers features:
    if os.path.isfile(path2SpeakerFeatures):
        speakerDatasetsFeatures = pickle.load(open(path2SpeakerFeatures, "rb"))
        # speakerDatasetsAudio = pickle.load(open(path2SpeakerAudio, "rb"))
    else:
        speakerDatasetsFeatures = createSpeakerWavs_Features(metadataSpeaker, fs, path2SpeakerAudio, path2SpeakerFeatures, 'speakers')

    # train speakers detection:
    if os.path.isfile(path2SpeakerModels):
        speakerModels = pickle.load(open(path2SpeakerModels, "rb"))
    else:
        speakerClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels)

if enableWordDetection:
    # create/load metadata:
    if os.path.isfile(path2metadataWordDict):
        metadataWord = pickle.load(open(path2metadataWordDict, "rb"))
    else:
        metadataWord = AudioMNISTMetaData()
        trainPortion, validatePortion, testPortion = 0.1, 0.1, 0.1
        metadataWord.label_train_sets(trainPortion, validatePortion, testPortion, genderEqual=True)
        pickle.dump(metadataWord, open(path2metadataWordDict, "wb"))

    fs = metadataWord.fs  # [hz]
    print('nMales, nFemales = %d, %d' % metadataWord.get_number_of_males_females())

    # create\load speakers features:
    if os.path.isfile(path2WordFeatures):
        wordDatasetsFeatures = pickle.load(open(path2WordFeatures, "rb"))
        # wordDatasetsAudio = pickle.load(open(path2WordAudio, "rb"))
        # sd.play(wordDatasetsAudio['train'][0][0],fs)
        # sd.stop()
    else:
        wordDatasetsFeatures = createSpeakerWavs_Features(metadataWord, fs, path2WordAudio, path2WordFeatures, 'words')

    # train speakers detection:
    if os.path.isfile(path2WordModels):
        wordModels = pickle.load(open(path2WordModels, "rb"))
    else:
        speakerClassificationTrain(wordDatasetsFeatures, path2WordModels, 'words')