from audioMNIST_mata2Dict import *
from speakerfeatures import *
import matplotlib.pyplot as plt
import random
import os
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM

random.seed(0)

path2metadataDict = './AudioMNISTMetadata.pt'
path2GenderFeatures = './GenderFeatures.pt'
path2GenderAudio = './GenderAudio.pt'
path2GenderModels = './GenderModels.pt'

nGenders = 2
femaleIdx, maleIdx = np.arange(nGenders)

# create/load metadata:
if os.path.isfile(path2metadataDict):
    metadata = pickle.load(open(path2metadataDict, "rb"))
else:
    metadata = AudioMNISTMetaData()
    trainPortion, validatePortion, testPortion = 0.1, 0.1, 0.1
    metadata.label_train_sets(trainPortion, validatePortion, testPortion)
    pickle.dump(metadata, open(path2metadataDict, "wb"))

fs = metadata.fs  # [hz]
print('nMales, nFemales = %d, %d' % metadata.get_number_of_males_females())

# create\load gender features:
if os.path.isfile(path2GenderFeatures):
    genderDatasetsFeatures = pickle.load(open(path2GenderFeatures, "rb"))
else:
    genderDatasetsAudio = dict()
    genderDatasetsFeatures = dict()
    for dataset in ['train', 'validate', 'test']:
        genderWavs = list(range(nGenders))
        if dataset == 'train':
            genderWavs[femaleIdx], genderWavs[maleIdx] = metadata.get_gender_train_set()
        elif dataset == 'validate':
            genderWavs[femaleIdx], genderWavs[maleIdx] = metadata.get_gender_validation_set()
        else:
            genderWavs[femaleIdx], genderWavs[maleIdx] = metadata.get_gender_test_set()

        genderAudio = list(range(len(genderWavs)))
        genderAudioLengths = list(range(len(genderWavs)))
        genderFeatures = list(range(len(genderWavs)))
        genderFeaturesLengths = list(range(len(genderWavs)))
        for genderIdx, singleGenderList in enumerate(genderWavs):
            for wavIdx, wav in enumerate(singleGenderList):
                if wavIdx%100 == 0:
                    print('dataset ',dataset,' starting gender %d feature extraction; wavIdx %d out of %d' % (genderIdx, wavIdx, len(singleGenderList)))
                extractedFeatures = np.float32(extract_features(wav, fs))
                wav = np.expand_dims(wav, axis=1)
                if wavIdx == 0:
                    genderAudio[genderIdx] = wav
                    genderAudioLengths[genderIdx] = list()
                    genderFeatures[genderIdx] = extractedFeatures
                    genderFeaturesLengths[genderIdx] = list()
                else:
                    genderAudio[genderIdx] = np.vstack((genderAudio[genderIdx], wav))
                    genderFeatures[genderIdx] = np.vstack((genderFeatures[genderIdx], extractedFeatures))
                genderAudioLengths[genderIdx].append(wav.shape[0])
                genderFeaturesLengths[genderIdx].append(extractedFeatures.shape[0])
        genderAudioList = [genderAudio, genderAudioLengths]
        genderFeaturesList = [genderFeatures, genderFeaturesLengths]
        genderDatasetsAudio[dataset] = genderAudioList
        genderDatasetsFeatures[dataset] = genderFeaturesList
    pickle.dump(genderDatasetsAudio, open(path2GenderAudio, "wb"))
    pickle.dump(genderDatasetsFeatures, open(path2GenderFeatures, "wb"))

# train gender detection:
if os.path.isfile(path2GenderModels):
    genderModels = pickle.load(open(path2GenderModels, "rb"))
else:
    covariance_type = 'diag'
    nTrainIters = 5
    max_nCorrect = -np.inf
    for trainIter in range(nTrainIters):
        genderModels = [GMMHMM(n_components=1, n_mix=6, n_iter=200, covariance_type=covariance_type, min_covar=1e-3) for genderIdx in range(nGenders)]
        for genderIdx in range(nGenders):
            genderModels[genderIdx].fit(genderDatasetsFeatures['train'][0][genderIdx], np.asarray(genderDatasetsFeatures['train'][1][genderIdx]))

        # validation:
        datasetKey = 'train'  # 'validate'
        nCorrect = 0
        nExamples = 0
        genderResults = list()
        for genderIdx in range(nGenders):
            nValExamples = np.asarray(genderDatasetsFeatures[datasetKey][1][genderIdx]).shape[0]
            genderResult = np.zeros((nValExamples, nGenders))
            startIdx = 0
            for singleValIdx in range(nValExamples):
                singleValLength = genderDatasetsFeatures[datasetKey][1][genderIdx][singleValIdx]
                stopIdx = startIdx + singleValLength
                for modelIdx in range(nGenders):
                    genderResult[singleValIdx, modelIdx] = genderModels[modelIdx].score(genderDatasetsFeatures[datasetKey][0][genderIdx][startIdx:stopIdx])
                startIdx += singleValLength
            predictedGender = np.argmax(genderResult, axis=1)
            nCorrectGender = (predictedGender == genderIdx).sum()
            nCorrect += nCorrectGender
            nExamples += nValExamples
            genderResults.append(genderResult) # for future use
            print('genderIdx: %d: %d correct out of %d' % (genderIdx, nCorrectGender, nValExamples))
        if nCorrect > max_nCorrect:
            max_nCorrect = nCorrect
            genderModels2Save = genderModels

    pickle.dump(genderModels2Save, open(path2GenderModels, "wb"))
