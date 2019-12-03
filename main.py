from audioMNIST_mata2Dict import *
from speakerfeatures import *
import matplotlib.pyplot as plt
import random
import os
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM
from sklearn.mixture import GaussianMixture

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
    #genderDatasetsAudio = pickle.load(open(path2GenderAudio, "rb"))
else:
    genderDatasetsFeatures = createGenderWavs_Features(metadata, fs, femaleIdx, maleIdx, path2GenderAudio, path2GenderFeatures)

# train gender detection:
if os.path.isfile(path2GenderModels):
    genderModels = pickle.load(open(path2GenderModels, "rb"))
else:
    genderClassificationTrain(genderDatasetsFeatures, path2GenderModels)
