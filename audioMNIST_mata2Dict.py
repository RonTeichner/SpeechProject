import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import mahalanobis as calcMahalanobis
from speakerfeatures import *
import matplotlib.pyplot as plt
import random
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM
from hmmlearn.utils import log_normalize
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from pysndfx import AudioEffectsChain
from copy import deepcopy
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import scipy.stats as stats

def createAudioMNIST_metadataDict():
    path2AudioMNIST = '../AudioMNIST/data/'
    audioMNIST_metadata = {
        "01": {
            "accent": "german",
            "age": 30,
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Wuerzburg",
            "recordingdate": "17-06-22-11-04-28",
            "recordingroom": "Kino"
        },
        "02": {
            "accent": "German",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Hamburg",
            "recordingdate": "17-06-26-17-57-29",
            "recordingroom": "Kino"
        },
        "03": {
            "accent": "German",
            "age": "31",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Bremen",
            "recordingdate": "17-06-30-17-34-51",
            "recordingroom": "Kino"
        },
        "04": {
            "accent": "German",
            "age": "23",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Helmstedt",
            "recordingdate": "17-06-30-18-09-14",
            "recordingroom": "Kino"
        },
        "05": {
            "accent": "German",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Hameln",
            "recordingdate": "17-07-06-10-53-10",
            "recordingroom": "Kino"
        },
        "06": {
            "accent": "German",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Dortmund",
            "recordingdate": "17-07-06-11-23-34",
            "recordingroom": "Kino"
        },
        "07": {
            "accent": "German/Spanish",
            "age": "27",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Spanien, Mallorca",
            "recordingdate": "17-07-10-17-06-17",
            "recordingroom": "Kino"
        },
        "08": {
            "accent": "German",
            "age": "41",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Ludwigsfelde",
            "recordingdate": "17-07-10-17-39-41",
            "recordingroom": "Kino"
        },
        "09": {
            "accent": "South Korean",
            "age": "35",
            "gender": "male",
            "native speaker": "no",
            "origin": "Asia, South Korea, Seoul",
            "recordingdate": "17-07-12-17-03-59",
            "recordingroom": "Kino"
        },
        "10": {
            "accent": "German",
            "age": "36",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Lemgo",
            "recordingdate": "17-07-12-17-31-43",
            "recordingroom": "Kino"
        },
        "11": {
            "accent": "German",
            "age": "33",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-07-12-17-59-59",
            "recordingroom": "Kino"
        },
        "12": {
            "accent": "German",
            "age": "26",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-07-19-17-05-31",
            "recordingroom": "Kino"
        },
        "13": {
            "accent": "German",
            "age": "27",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Freiberg",
            "recordingdate": "17-07-19-17-47-06",
            "recordingroom": "Kino"
        },
        "14": {
            "accent": "Spanish",
            "age": "31",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Spain, Oviedo",
            "recordingdate": "17-07-24-18-08-13",
            "recordingroom": "Kino"
        },
        "15": {
            "accent": "Madras",
            "age": "28",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, India, Chennai",
            "recordingdate": "17-07-24-19-06-38",
            "recordingroom": "Kino"
        },
        "16": {
            "accent": "German",
            "age": "30",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Muenchen",
            "recordingdate": "17-07-31-17-17-08",
            "recordingroom": "Kino"
        },
        "17": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-07-31-18-09-29",
            "recordingroom": "Kino"
        },
        "18": {
            "accent": "Levant",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Syria, Damascus",
            "recordingdate": "17-08-11-18-38-42",
            "recordingroom": "Kino"
        },
        "19": {
            "accent": "English",
            "age": "23",
            "gender": "male",
            "native speaker": "yes",
            "origin": "Europe, India, Delhi",
            "recordingdate": "17-08-11-19-12-12",
            "recordingroom": "Kino"
        },
        "20": {
            "accent": "German",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-08-14-17-27-02",
            "recordingroom": "Ruheraum"
        },
        "21": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Muenster",
            "recordingdate": "17-08-14-18-02-37",
            "recordingroom": "Ruheraum"
        },
        "22": {
            "accent": "German",
            "age": "33",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Braunschweig",
            "recordingdate": "17-08-15-13-38-24",
            "recordingroom": "Ruheraum"
        },
        "23": {
            "accent": "German",
            "age": "28",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Alsbach-Haehnlein",
            "recordingdate": "17-08-16-14-17-44",
            "recordingroom": "VR-Room"
        },
        "24": {
            "accent": "Chinese",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Asia, China, Nanning",
            "recordingdate": "17-08-16-15-37-02",
            "recordingroom": "VR-Room"
        },
        "25": {
            "accent": "Brasilian",
            "age": "22",
            "gender": "male",
            "native speaker": "no",
            "origin": "South-America, Brazil, Porto Alegre",
            "recordingdate": "17-08-16-16-01-05",
            "recordingroom": "VR-Room"
        },
        "26": {
            "accent": "Chinese",
            "age": "22",
            "gender": "female",
            "native speaker": "no",
            "origin": "Asia, China, Beijing",
            "recordingdate": "17-08-17-12-20-59",
            "recordingroom": "library"
        },
        "27": {
            "accent": "Italian",
            "age": "31",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Italy, Morbegno",
            "recordingdate": "17-08-17-12-50-32",
            "recordingroom": "library"
        },
        "28": {
            "accent": "German",
            "age": "28",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Hof",
            "recordingdate": "17-08-17-13-41-24",
            "recordingroom": "library"
        },
        "29": {
            "accent": "German",
            "age": "23",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Freiburg",
            "recordingdate": "17-09-14-20-54-30",
            "recordingroom": "vr-room"
        },
        "30": {
            "accent": "German",
            "age": "28",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Poland, Slubice",
            "recordingdate": "17-08-17-16-42-41",
            "recordingroom": "VR-Room"
        },
        "31": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-08-21-11-03-29",
            "recordingroom": "VR-room"
        },
        "32": {
            "accent": "Egyptian_American?",
            "age": "23",
            "gender": "male",
            "native speaker": "no",
            "origin": "Africa, Egypt, Alexandria",
            "recordingdate": "17-08-21-12-44-56",
            "recordingroom": "VR-room"
        },
        "33": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Hamburg",
            "recordingdate": "17-08-21-13-12-22",
            "recordingroom": "vr-room"
        },
        "34": {
            "accent": "German",
            "age": "25",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Munich",
            "recordingdate": "17-09-01-13-26-49",
            "recordingroom": "vr-room"
        },
        "35": {
            "accent": "Chinese",
            "age": "24",
            "gender": "male",
            "native speaker": "no",
            "origin": "Asia, China, Shanghai",
            "recordingdate": "17-08-21-15-03-12",
            "recordingroom": "vr-room"
        },
        "36": {
            "accent": "German",
            "age": "22",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-08-31-18-06-03",
            "recordingroom": "vr-room"
        },
        "37": {
            "accent": "Italian",
            "age": "27",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Italy, Casarsa",
            "recordingdate": "17-09-01-09-46-46",
            "recordingroom": "vr-room"
        },
        "38": {
            "accent": "Spanish",
            "age": "32",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Spain, Toledo",
            "recordingdate": "17-09-01-13-45-16",
            "recordingroom": "vr-room"
        },
        "39": {
            "accent": "German",
            "age": "29",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-01-14-13-15",
            "recordingroom": "vr-room"
        },
        "40": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-01-14-33-44",
            "recordingroom": "vr-room"
        },
        "41": {
            "accent": "South African",
            "age": "30",
            "gender": "male",
            "native speaker": "yes",
            "origin": "Africa, South Africa, Vryburg",
            "recordingdate": "17-09-01-14-56-32",
            "recordingroom": "vr-room"
        },
        "42": {
            "accent": "Arabic",
            "age": "29",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Syria, Damascus",
            "recordingdate": "17-09-01-15-23-24",
            "recordingroom": "vr-room"
        },
        "43": {
            "accent": "German",
            "age": "31",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Regensburg",
            "recordingdate": "17-09-01-16-59-50",
            "recordingroom": "vr-room"
        },
        "44": {
            "accent": "German",
            "age": "61",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-01-17-23-22",
            "recordingroom": "vr-room"
        },
        "45": {
            "accent": "German",
            "age": "1234",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Stuttgart",
            "recordingdate": "17-09-11-12-07-04",
            "recordingroom": "vr-room"
        },
        "46": {
            "accent": "German",
            "age": "30",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Vechta",
            "recordingdate": "17-09-11-13-59-04",
            "recordingroom": "vr-room"
        },
        "47": {
            "accent": "Danish",
            "age": "23",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Denmark, Copenhagen",
            "recordingdate": "17-09-11-14-33-03",
            "recordingroom": "vr-room"
        },
        "48": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-11-15-05-58",
            "recordingroom": "vr-room"
        },
        "49": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-12-14-50-32",
            "recordingroom": "vr-room"
        },
        "50": {
            "accent": "German",
            "age": "24",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Flensburg",
            "recordingdate": "17-09-12-18-25-00",
            "recordingroom": "vr-room"
        },
        "51": {
            "accent": "German",
            "age": "26",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Bremen",
            "recordingdate": "17-09-13-09-33-15",
            "recordingroom": "vr-room"
        },
        "52": {
            "accent": "French",
            "age": "34",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, France, Montpellier",
            "recordingdate": "17-09-13-10-32-26",
            "recordingroom": "vr-romm"
        },
        "53": {
            "accent": "German",
            "age": "24",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Reutlingen",
            "recordingdate": "17-09-13-10-58-47",
            "recordingroom": "vr-room"
        },
        "54": {
            "accent": "German",
            "age": "27",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-13-12-13-44",
            "recordingroom": "vr-room"
        },
        "55": {
            "accent": "German",
            "age": "23",
            "gender": "male",
            "native speaker": "no",
            "origin": "Europe, Germany, Dresden",
            "recordingdate": "17-09-13-12-35-54",
            "recordingroom": "vr-room"
        },
        "56": {
            "accent": "German",
            "age": "24",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Muenster",
            "recordingdate": "17-09-14-13-09-37",
            "recordingroom": "vr-room"
        },
        "57": {
            "accent": "German",
            "age": "27",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-09-15-13-21-33",
            "recordingroom": "vr-room"
        },
        "58": {
            "accent": "German",
            "age": "29",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-10-19-20-35-42",
            "recordingroom": "vr-room"
        },
        "59": {
            "accent": "German",
            "age": "31",
            "gender": "female",
            "native speaker": "no",
            "origin": "Europe, Germany, Berlin",
            "recordingdate": "17-10-19-21-03-53",
            "recordingroom": "vr-room"
        },
        "60": {
            "accent": "Tamil",
            "age": "27",
            "gender": "female",
            "native speaker": "yes",
            "origin": "Asia, India, Chennai",
            "recordingdate": "17-10-20-17-24-39",
            "recordingroom": "vr-room"
        }
    }
    includedInTrainSet = None
    for key in audioMNIST_metadata.keys():
        path2Library = path2AudioMNIST + key
        listOfFiles = os.listdir(path2Library)
        numbersDict = dict()
        for digit in range(10):
            numbersDict[digit] = list() # list of tuples
        for fileName in listOfFiles:
            spokenDigit = int(fileName.split('_')[0])
            numbersDict[spokenDigit].append((path2Library + '/' + fileName, includedInTrainSet))
        audioMNIST_metadata[key]['pathsDict'] = numbersDict
    return audioMNIST_metadata


class AudioMNISTMetaData:
    def __init__(self):
        self.metaDataDict = createAudioMNIST_metadataDict()
        self.trainEnum, self.validateEnum, self.testEnum, self.noneEnum = 0, 1, 2, 3
        for libraryKey in self.metaDataDict.keys():
            for digitKey in self.metaDataDict[libraryKey]['pathsDict'].keys():
                for pathIdx in range(len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])):
                    self.fs, _ = wavfile.read(self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][0])
                    break
                break
            break

    def label_train_sets(self, trainPortion, validatePortion, testPortion, genderEqual):
        nonePortion = 1 - (trainPortion + validatePortion + testPortion)
        if genderEqual:
            nMales, nFemales = self.get_number_of_males_females()
            # We want an equal no. of men and women files within the sets
            female2maleFactor = nFemales/nMales
            if female2maleFactor < 1:
                femaleTrainPortion, femaleValidationPortion, femaleTestPortion = trainPortion/female2maleFactor, validatePortion/female2maleFactor, testPortion/female2maleFactor
                if femaleTrainPortion + femaleValidationPortion + femaleTestPortion > 1:
                    shrinkFactor = femaleTrainPortion + femaleValidationPortion + femaleTestPortion
                    femaleTrainPortion, femaleValidationPortion, femaleTestPortion, femaleNonePortion = femaleTrainPortion/shrinkFactor, femaleValidationPortion/shrinkFactor, femaleTestPortion/shrinkFactor, 0
                    maleTrainPortion, maleValidationPortion, maleTestPortion = trainPortion / shrinkFactor, validatePortion / shrinkFactor, testPortion / shrinkFactor
                    maleNonePortion = 1 - (maleTrainPortion + maleValidationPortion + maleTestPortion)
                else:
                    maleTrainPortion, maleValidationPortion, maleTestPortion, maleNonePortion = trainPortion, validatePortion, testPortion, nonePortion
                    femaleNonePortion = 1 - (femaleTrainPortion + femaleValidationPortion + femaleTestPortion)
            else:
                raise Exception('dataset with more females than males, not implemeted')
        else:
            maleTrainPortion, maleValidationPortion, maleTestPortion, maleNonePortion = trainPortion, validatePortion, testPortion, nonePortion
            femaleTrainPortion, femaleValidationPortion, femaleTestPortion, femaleNonePortion = trainPortion, validatePortion, testPortion, nonePortion


        for libraryKey in self.metaDataDict.keys():
            for digitKey in self.metaDataDict[libraryKey]['pathsDict'].keys():
                for pathIdx in range(len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])):
                    y = list(self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx])
                    if self.metaDataDict[libraryKey]['gender'] == 'male':
                        y[1] = int(np.argwhere(np.random.multinomial(1, pvals=[maleTrainPortion, maleValidationPortion, maleTestPortion, maleNonePortion])))
                    else:
                        y[1] = int(np.argwhere(np.random.multinomial(1, pvals=[femaleTrainPortion, femaleValidationPortion, femaleTestPortion, femaleNonePortion])))
                    self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx] = tuple(y)

    def get_gender_set(self, dataset):
        if dataset == 'train':
            datasetEnum = self.trainEnum
        elif dataset == 'validate':
            datasetEnum = self.validateEnum
        else:
            datasetEnum = self.testEnum
        femaleList, maleList = list(), list()
        for libraryKey in self.metaDataDict.keys():
            for digitKey in self.metaDataDict[libraryKey]['pathsDict'].keys():
                for pathIdx in range(len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])):
                    included = self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][1] == datasetEnum
                    if included:
                        _, wavData = wavfile.read(self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][0])
                        if self.metaDataDict[libraryKey]['gender'] == 'male':
                            maleList.append(wavData)
                        else:
                            femaleList.append(wavData)
        return [femaleList, maleList]

    def get_speakers_gender_list(self):
        speakersGenderList = list()
        for libraryKey in self.metaDataDict.keys():
            speakersGenderList.append(self.metaDataDict[libraryKey]['gender'])
        return speakersGenderList

    def get_speaker_set(self, dataset):
        if dataset == 'train':
            datasetEnum = self.trainEnum
        elif dataset == 'validate':
            datasetEnum = self.validateEnum
        else:
            datasetEnum = self.testEnum

        speakersLists = list()
        for libraryKey in self.metaDataDict.keys():
            speakersLists.append(list())
            for digitKey in self.metaDataDict[libraryKey]['pathsDict'].keys():
                for pathIdx in range(len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])):
                    included = self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][1] == datasetEnum
                    if included:
                        _, wavData = wavfile.read(self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][0])
                        speakersLists[-1].append(wavData)
        return speakersLists

    def get_word_set(self, dataset):
        if dataset == 'train':
            datasetEnum = self.trainEnum
        elif dataset == 'validate':
            datasetEnum = self.validateEnum
        else:
            datasetEnum = self.testEnum

        wordLists = list()
        firstLibraryKey = [*self.metaDataDict.keys()][0]
        for digitKey in self.metaDataDict[firstLibraryKey]['pathsDict'].keys(): # digitKeys from all libraries are the same
            wordLists.append(list())
            for libraryKey in self.metaDataDict.keys():
                for pathIdx in range(len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])):
                    included = self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][1] == datasetEnum
                    if included:
                        _, wavData = wavfile.read(self.metaDataDict[libraryKey]['pathsDict'][digitKey][pathIdx][0])
                        wordLists[-1].append(wavData)
        return wordLists

    def get_number_of_males_females(self):
        nMales, nFemales = 0, 0
        for libraryKey in self.metaDataDict.keys():
            for digitKey in self.metaDataDict[libraryKey]['pathsDict'].keys():
                nFilesInDir = len(self.metaDataDict[libraryKey]['pathsDict'][digitKey])
                if self.metaDataDict[libraryKey]['gender'] == 'male':
                    nMales += nFilesInDir
                else:
                    nFemales += nFilesInDir
        return nMales, nFemales

def createcategoryWavs_Features(metadata, fs, path2categoryAudio, path2categoryFeatures, type):
    categoryDatasetsAudio = dict()
    categoryDatasetsFeatures = dict()
    for dataset in ['train', 'validate', 'test']:
        if type == 'speakers':
            categoryWavs = metadata.get_speaker_set(dataset)
        elif type == 'genders':
            categoryWavs = metadata.get_gender_set(dataset)
        elif type == 'words':
            categoryWavs = metadata.get_word_set(dataset)

        categoryAudio = list(range(len(categoryWavs)))
        categoryAudioLengths = list(range(len(categoryWavs)))
        categoryFeatures = list(range(len(categoryWavs)))
        categoryFeaturesLengths = list(range(len(categoryWavs)))
        for categoryIdx, singlecategoryList in enumerate(categoryWavs):
            for wavIdx, wav in enumerate(singlecategoryList):
                if wavIdx%100 == 0:
                    print('createcategoryWavs_Features: dataset ',dataset,' starting category %d feature extraction; wavIdx %d out of %d' % (categoryIdx, wavIdx, len(singlecategoryList)))
                extractedFeatures = np.float32(extract_features(wav, fs))
                wav = np.expand_dims(wav, axis=1)
                if wavIdx == 0:
                    categoryAudio[categoryIdx] = wav
                    categoryAudioLengths[categoryIdx] = list()
                    categoryFeatures[categoryIdx] = extractedFeatures
                    categoryFeaturesLengths[categoryIdx] = list()
                else:
                    categoryAudio[categoryIdx] = np.vstack((categoryAudio[categoryIdx], wav))
                    categoryFeatures[categoryIdx] = np.vstack((categoryFeatures[categoryIdx], extractedFeatures))
                categoryAudioLengths[categoryIdx].append(wav.shape[0])
                categoryFeaturesLengths[categoryIdx].append(extractedFeatures.shape[0])
        categoryAudioList = [categoryAudio, categoryAudioLengths]
        categoryFeaturesList = [categoryFeatures, categoryFeaturesLengths]
        categoryDatasetsAudio[dataset] = categoryAudioList
        categoryDatasetsFeatures[dataset] = categoryFeaturesList
    pickle.dump(categoryDatasetsAudio, open(path2categoryAudio, "wb"))
    pickle.dump(categoryDatasetsFeatures, open(path2categoryFeatures, "wb"))
    return categoryDatasetsFeatures


def categoryClassificationTrain(categoryDatasetsFeatures, path2categoryModels, type='speaker', trainOnLessFeatures=False, chosenFeatures=[]):
    ncategorys = len(categoryDatasetsFeatures['train'][0])
    covariance_type = 'diag'
    nTrainIters = 1
    max_nCorrect = -np.inf
    for trainIter in range(nTrainIters):
        if type == 'speaker':
            nHmmStates, nMix = 1, 3
        elif type == 'words':
            nHmmStates, nMix = 1, 1
        categoryModels = [GMMHMM(n_components=nHmmStates, n_mix=nMix, n_iter=200, covariance_type=covariance_type) for categoryIdx in range(ncategorys)]
        for categoryIdx in range(ncategorys):
            lengthsVec = np.asarray(categoryDatasetsFeatures['train'][1][categoryIdx])
            if nHmmStates == 1:
                # empirically, train results are better with all the gender signals collapsed to a single signal
                lengthsVec = np.expand_dims(lengthsVec.sum(), axis=0)
                allFeaturesAllFrames = categoryDatasetsFeatures['train'][0][categoryIdx]
            if trainOnLessFeatures:
                allFeaturesAllFrames = limitFeatures(allFeaturesAllFrames, chosenFeatures)
            categoryModels[categoryIdx].fit(allFeaturesAllFrames, lengths=lengthsVec)

        # validation:
        datasetKey = 'validate'
        nCorrect = 0
        nExamples = 0
        genderResults = list()
        for categoryIdx in range(ncategorys):
            nValExamples = np.asarray(categoryDatasetsFeatures[datasetKey][1][categoryIdx]).shape[0]
            categoryResult = np.zeros((nValExamples, ncategorys))
            startIdx = 0
            for singleValIdx in range(nValExamples):
                singleValLength = categoryDatasetsFeatures[datasetKey][1][categoryIdx][singleValIdx]
                stopIdx = startIdx + singleValLength
                wavFeatures = categoryDatasetsFeatures[datasetKey][0][categoryIdx][startIdx:stopIdx]
                if trainOnLessFeatures:
                    wavFeatures = limitFeatures(wavFeatures, chosenFeatures)
                for modelIdx in range(ncategorys):
                    categoryResult[singleValIdx, modelIdx] = categoryModels[modelIdx].score(wavFeatures)
                startIdx += singleValLength
            predictedcategory = np.argmax(categoryResult, axis=1)
            nCorrectcategory = (predictedcategory == categoryIdx).sum()
            nCorrect += nCorrectcategory
            nExamples += nValExamples
            genderResults.append(categoryResult) # for future use
            print('categoryClassificationTrain: hmmgmm: groupIdx: %d: %d correct out of %d <=> %02.0f%%' % (categoryIdx, nCorrectcategory, nValExamples, nCorrectcategory/nValExamples*100))
        if nCorrect > max_nCorrect:
            max_nCorrect = nCorrect
            categoryModels2Save = categoryModels
        print('categoryClassificationTrain: best performance on validation set: %d correct out of %d <=> %02.0f%%' % (nCorrect, nExamples, nCorrect / nExamples * 100))

    pickle.dump(categoryModels2Save, open(path2categoryModels, "wb"))
    return categoryModels2Save

def createSentencesMetadata(metadata, path2SentencesMetadata, nSentences=500, whichSet='test'):
    # create sentence transitionMat:
    nDigits = 10
    transitionMat = np.zeros((nDigits, nDigits))
    for currentState in range(nDigits):
        for nextState in range(nDigits):
            if nextState == currentState:
                i = nDigits
            else:
                i = (nextState - currentState) % nDigits
            transitionMat[currentState, nextState] = np.exp(-i) # np.power(2.0, -i)

    for currentState in range(nDigits):
        transitionMat[currentState, :] = transitionMat[currentState, :] / transitionMat[currentState, :].sum()

    priorStates = np.ones(nDigits) / nDigits
    # The dataset will consist of a list of sublists. each sublist has a library key (pointing to a specific speaker) in the first entrie.
    # Following are tuples - digitKey & path to specific file

    min_nWordsPerSentence = 3
    max_nWordsPerSentence = 8
    listOfLibraryKeys = list(metadata.metaDataDict.keys())
    sentencesMetadata = list()
    for sentenceIdx in range(nSentences):
        print('createSentencesMetadata: starting sentence no. %d out of %d' % (sentenceIdx, nSentences))
        sentencesMetadata.append(list())
        nSearchIters = 1000
        sentenceFound = False
        while not sentenceFound:
            specificSentence = list()
            searchIter = 0
            libraryKey = random.choice(listOfLibraryKeys)
            specificSentence.append(libraryKey)
            sentenceLength = np.random.randint(low=min_nWordsPerSentence, high=max_nWordsPerSentence + 1)
            forBreakFlag = False
            for digitIdx in range(sentenceLength):
                if digitIdx == 0:
                    digit = int(np.argwhere(np.random.multinomial(1, pvals=priorStates)))
                else:
                    digit = int(np.argwhere(np.random.multinomial(1, pvals=transitionMat[digit, :])))
                foundTest = False
                while not foundTest:
                    specificDigit = list(random.choice(metadata.metaDataDict[libraryKey]['pathsDict'][digit]))
                    if whichSet == 'test':
                        foundTest = specificDigit[1] == metadata.testEnum
                    elif whichSet == 'train':
                        foundTest = specificDigit[1] == metadata.trainEnum
                    elif whichSet == 'validate':
                        foundTest = specificDigit[1] == metadata.validateEnum
                    #if not(foundTest):
                        #print('digit = %d, sentenceIdx = %d, digitIdx = %d; testEnum not found' % (digit, sentenceIdx, digitIdx))
                    searchIter += 1
                    if searchIter > nSearchIters:
                        forBreakFlag = True
                        break
                if foundTest: specificSentence.append((digit, specificDigit[0]))
                if digitIdx == (sentenceLength-1) and foundTest: sentenceFound=True
                if forBreakFlag: break
        sentencesMetadata[-1] = specificSentence
    pickle.dump([sentencesMetadata, priorStates, transitionMat], open(path2SentencesMetadata, "wb"))
    return sentencesMetadata, priorStates, transitionMat

def createSentenceWavs_Features(sentencesMetadata, path2SentencesAudio, path2SentencesFeatures, includeEffects=False, createPitch=False):
    if includeEffects:
        fx = (AudioEffectsChain().reverb(room_scale=100, wet_gain=10, hf_damping=100, reverberance=100))

    sentencesAudio = deepcopy(sentencesMetadata)
    sentencesFeatures = deepcopy(sentencesMetadata)
    nPitchErrors = 0
    for sentenceIdx, sentenceMetadata in enumerate(sentencesMetadata):
        if sentenceIdx % 100 == 0:
            print('createSentenceWavs_Features: starting sentence %d out of %d' % (sentenceIdx, len(sentencesMetadata)))
        for wordIdx, wordPath in enumerate(sentenceMetadata):
            if wordIdx == 0: continue
            digit, filename = wordPath[0], list(wordPath)[1]
            if not(createPitch):
                fs, wav = wavfile.read(filename)
                if includeEffects:
                    wav = fx(wav)

                sentencesAudio[sentenceIdx][wordIdx] = (digit, wav)
                sentencesFeatures[sentenceIdx][wordIdx] = (digit, np.float32(extract_features(wav, fs)))
            else:
                signal = basic.SignalObj(filename)
                try:
                    pitch = pYAAPT.yaapt(signal)
                    sentencesFeatures[sentenceIdx][wordIdx] = (digit, np.float32(pitch.samp_values))
                except:
                    sentencesFeatures[sentenceIdx][wordIdx] = (digit, -1.0)
                    nPitchErrors += 1
                    print('createSentenceWavs_Features: no. of pitch errors: %d' % nPitchErrors)
    print('createSentenceWavs_Features: no. of pitch errors: %d' % nPitchErrors)
    if path2SentencesAudio is not None: pickle.dump(sentencesAudio, open(path2SentencesAudio, "wb"))
    pickle.dump(sentencesFeatures, open(path2SentencesFeatures, "wb"))
    return sentencesFeatures

def log2probs(logProbs):
    logProbs -= logProbs.max() # now max value is zero
    probs = np.exp(np.maximum(logProbs, -30)) # setting min value of -30 for log-values
    return probs / probs.sum()

def computeFilteringSmoothing(models, sentence, sentenceModel, trainOnLessFeatures=False, enableMahalanobisCala=False, chosenFeatures=[]):
    nModels, nWords = len(models), len(sentence)-1
    wordlogprob = np.zeros([nWords, nModels])
    mahalanobis = list()
    for wordIdx in range(nWords):
        wordFeatures = sentence[wordIdx + 1][1]
        if trainOnLessFeatures:
            wordFeatures = limitFeatures(wordFeatures, chosenFeatures)
        for modelIdx in range(nModels):
            wordlogprob[wordIdx, modelIdx] = models[modelIdx].score(wordFeatures)
        if enableMahalanobisCala:
            nFrames, nFeatures, nMix = wordFeatures.shape[0], models[0].means_.shape[2], models[0].means_.shape[1]
            if nMix == 1:
                mahalanobisDist = np.zeros([nFrames, nModels, nFeatures])
                for frameIdx in range(nFrames):
                    for modelIdx in range(nModels):
                        for featureIdx in range(nFeatures):
                            mahalanobisDist[frameIdx, modelIdx, featureIdx] = calcMahalanobis(wordFeatures[frameIdx, featureIdx], models[modelIdx].means_[0, 0, featureIdx], 1/models[modelIdx].covars_[0, 0, featureIdx])
                mahalanobis.append(mahalanobisDist)


    # probs = np.apply_along_axis(log2probs, axis=1, arr=wordlogprob.copy())
    logprob, fwdlattice = sentenceModel._do_forward_pass(wordlogprob)
    bwdlattice = sentenceModel._do_backward_pass(wordlogprob)

    # log_gamma = fwdlattice + bwdlattice
    # logSmoothing = log_gamma.copy()

    # log_normalize(logSmoothing, axis=1)

    smoothing = sentenceModel._compute_posteriors(fwdlattice, bwdlattice)
    # filtering = np.apply_along_axis(log2probs, axis=1, arr=fwdlattice.copy())
    logFiltering = fwdlattice.copy()
    log_normalize(logFiltering, axis=1)
    with np.errstate(under="ignore"):
        filtering = np.exp(logFiltering)

    log_normalize(wordlogprob, axis=1)
    #with np.errstate(under="ignore"):
        #rawFrameProb = np.exp(framelogprob)
    '''
    # code for verifying that the transmat is read correctly by self implementation of the forward pass:
    probs = np.random.rand(*framelogprob.shape)
    myfwdlattice = myForwardPass(np.log(probs), sentenceModel.transmat_)
    _, myfwdlatticeCompare = sentenceModel._do_forward_pass(np.log(probs))
    compRes = myfwdlatticeCompare - myfwdlattice
    '''
    return filtering, smoothing, mahalanobis

def myForwardPass(framelogprob, transmat):
    myfwdlattice = np.zeros_like(framelogprob)
    nFrames = framelogprob.shape[0]
    nStates = framelogprob.shape[1]
    myfwdlattice[0] = framelogprob[0] + np.log(1/nStates)
    # transmat = transmat.transpose()
    for latticeFrameIdx in range(1,nFrames):
        for nextState in range(nStates):
            currentWordProbs = np.exp(myfwdlattice[latticeFrameIdx - 1])
            priorProbOfNextState = (currentWordProbs * transmat[:, nextState]).sum()
            posteriorProbOfNextState = priorProbOfNextState * np.exp(framelogprob[latticeFrameIdx, nextState])
            myfwdlattice[latticeFrameIdx, nextState] = np.log(posteriorProbOfNextState)
    return  myfwdlattice

def createSentencesEstimationResults(sentencesDatasetsFeatures, sentencesDatasetsPitch, metadata, path2SentencesResults, path2WordModels, path2SpeakerModels, path2GenderModels, transitionMat, priorStates, trainOnLessFeatures=False, enableMahalanobisCala=False, chosenFeatures=[]):
    # load models:
    wordModels = pickle.load(open(path2WordModels, "rb"))
    speakerModels = pickle.load(open(path2SpeakerModels, "rb"))
    genderModels = pickle.load(open(path2GenderModels, "rb"))

    nGenderModels = len(genderModels)
    genderSentenceModel = GMMHMM(n_components=nGenderModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nGenderModels))  # fit creates all internal variables
    genderSentenceModel.transmat_, genderSentenceModel.startprob_ = np.eye(nGenderModels), np.ones(nGenderModels) / nGenderModels

    nSpeakersModels = len(speakerModels)
    speakerSentenceModel = GMMHMM(n_components=nSpeakersModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nSpeakersModels))  # fit creates all internal variables
    speakerSentenceModel.transmat_, speakerSentenceModel.startprob_ = np.eye(nSpeakersModels), np.ones(nSpeakersModels) / nSpeakersModels

    nWordModels = len(wordModels)
    wordSentenceModel = GMMHMM(n_components=nWordModels, n_mix=1, n_iter=200, covariance_type='diag').fit(np.random.randn(100, nWordModels))  # fit creates all internal variables
    wordSentenceModel.transmat_, wordSentenceModel.startprob_ = transitionMat, priorStates

    # compute model estimations:
    sentencesEstimationResults = list()
    for sentenceIdx, sentence in enumerate(sentencesDatasetsFeatures):
        if sentenceIdx % 100 == 0:
            print('createSentencesEstimationResults: starting sentence %d estimation out of %d' % (sentenceIdx, len(sentencesDatasetsFeatures)))
        sentenceDict = dict()
        nWords = len(sentence)
        sentenceDict['groundTruth'] = dict()
        sentenceDict['groundTruth']['SpeakerNo'] = sentence[0]
        sentenceDict['groundTruth']['SpeakerGender'] = metadata.metaDataDict[sentence[0]]['gender']
        sentenceDict['groundTruth']['Digits'] = [sentence[i][0] for i in range(1, nWords)]

        sentenceDict['results'] = dict()

        sentenceDict['results']['gender'] = dict()
        sentenceDict['results']['gender']['filtering'], sentenceDict['results']['gender']['smoothing'], sentenceDict['results']['gender']['mahalanobis'] = computeFilteringSmoothing(genderModels, sentence, genderSentenceModel, trainOnLessFeatures=trainOnLessFeatures, enableMahalanobisCala=False, chosenFeatures=chosenFeatures)

        sentenceDict['results']['speaker'] = dict()
        sentenceDict['results']['speaker']['filtering'], sentenceDict['results']['speaker']['smoothing'], sentenceDict['results']['speaker']['mahalanobis'] = computeFilteringSmoothing(speakerModels, sentence, speakerSentenceModel, trainOnLessFeatures=trainOnLessFeatures, enableMahalanobisCala=False, chosenFeatures=chosenFeatures)

        sentenceDict['results']['word'] = dict()
        sentenceDict['results']['word']['filtering'], sentenceDict['results']['word']['smoothing'], sentenceDict['results']['word']['mahalanobis'] = computeFilteringSmoothing(wordModels, sentence, wordSentenceModel, trainOnLessFeatures=trainOnLessFeatures, enableMahalanobisCala=False, chosenFeatures=chosenFeatures)

        sentenceDict['results']['pitch'] = dict()
        sentenceDict['results']['pitch']['filtering'], sentenceDict['results']['pitch']['smoothing'] = computePitchDistribution(sentencesDatasetsPitch[sentenceIdx])

        sentencesEstimationResults.append(sentenceDict)
    pickle.dump(sentencesEstimationResults, open(path2SentencesResults, "wb"))
    return sentencesEstimationResults

def computePitchDistribution(sentence):
    n_components = 2 # from impression, 2 is the correct no.
    filteringMeans = np.zeros((n_components, len(sentence)-1))
    filteringCovs = np.zeros((n_components, len(sentence) - 1))
    filteringWeights = np.zeros((n_components, len(sentence)-1))
    pitchValuesWereExtracted = False
    for wordIdx, word in enumerate(sentence):
        if wordIdx == 0:
            speakerNo = int(word)
            continue
        if np.max(word[1]) == -1:
            continue  # no pitch values were extracted
        pitchIndexes = word[1].nonzero()
        pitchValues = word[1][pitchIndexes]
        if not pitchValuesWereExtracted:
            pitchValuesWereExtracted = True
            allPitchValues = pitchValues
        else:
            allPitchValues = np.hstack((allPitchValues, pitchValues))
        GaussianMixModel = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar=1e-1).fit(np.expand_dims(allPitchValues, axis=1))
        filteringMeans[:, wordIdx-1:wordIdx], filteringCovs[:, wordIdx-1:wordIdx], filteringWeights[:, wordIdx-1:wordIdx] = GaussianMixModel.means_, GaussianMixModel.covariances_, np.expand_dims(GaussianMixModel.weights_, axis=1)
    filteringDict = dict()
    filteringDict['means'], filteringDict['covs'], filteringDict['weights'] = filteringMeans, filteringCovs, filteringWeights
    smoothingDict = dict()
    smoothingDict['means'], smoothingDict['covs'], smoothingDict['weights'] = filteringMeans[:, -1], filteringCovs[:, -1], filteringWeights[:, -1]
    if False:
        n_bins = 20
        n, bins, patches = plt.hist(allPitchValues, n_bins, density=True, histtype='step', cumulative=False, label='hist')
        for componentIdx in range(n_components):
            mu, sigma, weight = smoothingDict['means'][componentIdx], np.sqrt(smoothingDict['covs'][componentIdx]), smoothingDict['weights'][componentIdx]
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x, weight * stats.norm.pdf(x, mu, sigma), label='component %d' % componentIdx)
        plt.xlabel('Pitch [Hz]')
        plt.legend()
        # plt.xlim(50, 400)
        plt.show()
    return filteringDict, smoothingDict

def plotSentenceResults(sentencesEstimationResults, maleIdx, femaleIdx, path2fig, sentencesEstimationResults_NN=None):
    # convert model first-word estimations to np-arrays:
    genderIdx_NN, speakerIdx_NN, wordIdx_NN, pitchIdx_NN = np.arange(4)
    nSentences = len(sentencesEstimationResults)
    classCategories = ['word', 'gender', 'speaker', 'pitch']
    collectedFirstWordSentenceResults = dict()
    for estimationClass in classCategories:
        collectedFirstWordSentenceResults[estimationClass] = dict()
        firstDigit_filtering, firstDigit_smoothing, firstDigit_INPV = np.zeros(nSentences), np.zeros(nSentences), np.zeros(nSentences)
        for sentenceIdx in range(nSentences):
            sentenceResult = sentencesEstimationResults[sentenceIdx]
            if sentencesEstimationResults_NN is not None: weights_NN, LUT_NN = sentencesEstimationResults_NN[sentenceIdx]
            if estimationClass == 'word':
                trueFirstDigit = sentenceResult['groundTruth']['Digits'][0]
                firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueFirstDigit]
                firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueFirstDigit]
                if sentencesEstimationResults_NN is not None:
                    trueValIndexesInLUT = np.where(LUT_NN[:, wordIdx_NN] == trueFirstDigit)[0]
                    firstDigit_INPV[sentenceIdx] = np.sum([weights_NN[i] for i in trueValIndexesInLUT])
            elif estimationClass == 'gender':
                trueGender = sentenceResult['groundTruth']['SpeakerGender']
                if trueGender == 'male':
                    trueGenderIdx = maleIdx
                else:
                    trueGenderIdx = femaleIdx
                firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueGenderIdx]
                firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueGenderIdx]
                if sentencesEstimationResults_NN is not None:
                    trueValIndexesInLUT = np.where(LUT_NN[:, genderIdx_NN] == trueGenderIdx)[0]
                    firstDigit_INPV[sentenceIdx] = np.sum([weights_NN[i] for i in trueValIndexesInLUT])
            elif estimationClass == 'speaker':
                trueSpeakerNo = int(sentenceResult['groundTruth']['SpeakerNo']) - 1
                firstDigit_filtering[sentenceIdx] = sentenceResult['results'][estimationClass]['filtering'][0][trueSpeakerNo]
                firstDigit_smoothing[sentenceIdx] = sentenceResult['results'][estimationClass]['smoothing'][0][trueSpeakerNo]
                if sentencesEstimationResults_NN is not None:
                    trueValIndexesInLUT = np.where(LUT_NN[:, speakerIdx_NN] == trueSpeakerNo)[0]
                    firstDigit_INPV[sentenceIdx] = np.sum([weights_NN[i] for i in trueValIndexesInLUT])
            elif estimationClass == 'pitch':
                firstDigit_filtering[sentenceIdx] = calcStdOfMixOf2(sentenceResult['results']['pitch']['filtering']['means'][:, 0], sentenceResult['results']['pitch']['filtering']['covs'][:, 0], sentenceResult['results']['pitch']['filtering']['weights'][:, 0])
                firstDigit_smoothing[sentenceIdx] = calcStdOfMixOf2(sentenceResult['results']['pitch']['filtering']['means'][:, -1], sentenceResult['results']['pitch']['filtering']['covs'][:, -1], sentenceResult['results']['pitch']['filtering']['weights'][:, -1])
                if sentencesEstimationResults_NN is not None:
                    pitchValues_NN = LUT_NN[:, pitchIdx_NN]
                    pitchMean_NN = pitchValues_NN.mean()
                    pitchVar = np.sum(np.multiply(weights_NN, np.power(pitchValues_NN-pitchMean_NN, 2)))
                    firstDigit_INPV[sentenceIdx] = np.sqrt(pitchVar)
        collectedFirstWordSentenceResults[estimationClass]['filtering'] = firstDigit_filtering
        collectedFirstWordSentenceResults[estimationClass]['smoothing'] = firstDigit_smoothing
        if sentencesEstimationResults_NN is not None:
            collectedFirstWordSentenceResults[estimationClass]['INPV'] = firstDigit_INPV

    # plot first-word estimation CDFs of true value:
    fig = plt.subplots(figsize=(24, 10))
    plt.suptitle('Likelihoods - CDF & Histograms', fontsize=16)
    for plotIdx, estimationClass in enumerate(classCategories):
        plt.subplot(2, len(classCategories), plotIdx + 1)
        n_bins = 100
        #n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['rawFrameMahalanobis'], n_bins, density=True, histtype='step', cumulative=True, label='Raw')
        n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['filtering'], n_bins, density=True, histtype='step', cumulative=True, label='Filtering')
        n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['smoothing'], n_bins, density=True, histtype='step', cumulative=True, label='Smoothing')
        if sentencesEstimationResults_NN is not None:
            n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['INPV'], n_bins, density=True, histtype='step', cumulative=True, label='INPV')
        #meanRaw = collectedFirstWordSentenceResults[estimationClass]['rawFrameMahalanobis'].mean()
        meanFiltering = collectedFirstWordSentenceResults[estimationClass]['filtering'].mean()
        meanSmoothing = collectedFirstWordSentenceResults[estimationClass]['smoothing'].mean()
        if sentencesEstimationResults_NN is not None:
            meanINPV = collectedFirstWordSentenceResults[estimationClass]['INPV'].mean()
        plt.grid(True)
        plt.legend(loc='right')
        #plt.title(estimationClass + ' likelihood CDF; avg(R,F,S) = (%02.0f%%,%02.0f%%,%02.0f%%)' % (meanRaw * 100, meanFiltering * 100, meanSmoothing * 100))
        if sentencesEstimationResults_NN is not None:
            if estimationClass == 'pitch':
                plt.title(estimationClass + ' std avg(F,S) = (%02.0f,%02.0f,%02.0f) Hz' % (meanFiltering, meanSmoothing, meanINPV))
                plt.xlabel('Hz')
            else:
                plt.title(estimationClass + ' avg(F,S) = (%02.0f%%,%02.0f%%,%02.0f%%)' % (meanFiltering * 100, meanSmoothing * 100, meanINPV * 100))
                plt.xlabel('likelihood')
        else:
            if estimationClass == 'pitch':
                plt.title(estimationClass + ' std avg(F,S) = (%02.0f,%02.0f) Hz' % (meanFiltering, meanSmoothing))
                plt.xlabel('Hz')
            else:
                plt.title(estimationClass + ' avg(F,S) = (%02.0f%%,%02.0f%%)' % (meanFiltering * 100, meanSmoothing * 100))
                plt.xlabel('likelihood')

        plt.subplot(2, len(classCategories), plotIdx + 1 + len(classCategories))
        n_bins = 20
        # n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['rawFrameMahalanobis'], n_bins, density=True, histtype='step', cumulative=True, label='Raw')
        n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['filtering'], n_bins, density=True, histtype='step', cumulative=False, label='Filtering')
        n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['smoothing'], n_bins, density=True, histtype='step', cumulative=False, label='Smoothing')
        if sentencesEstimationResults_NN is not None:
            n, bins, patches = plt.hist(collectedFirstWordSentenceResults[estimationClass]['INPV'], n_bins, density=True, histtype='step', cumulative=False, label='INPV')
        # meanRaw = collectedFirstWordSentenceResults[estimationClass]['rawFrameMahalanobis'].mean()
        meanFiltering = collectedFirstWordSentenceResults[estimationClass]['filtering'].mean()
        meanSmoothing = collectedFirstWordSentenceResults[estimationClass]['smoothing'].mean()
        if sentencesEstimationResults_NN is not None:
            meanINPV = collectedFirstWordSentenceResults[estimationClass]['INPV'].mean()
        plt.grid(True)
        plt.legend(loc='right')
        # plt.title(estimationClass + ' likelihood CDF; avg(R,F,S) = (%02.0f%%,%02.0f%%,%02.0f%%)' % (meanRaw * 100, meanFiltering * 100, meanSmoothing * 100))
        if sentencesEstimationResults_NN is not None:
            if estimationClass == 'pitch':
                #plt.title(estimationClass + ' std histogram; avg(F,S) = (%02.0f,%02.0f,%02.0f) Hz' % (meanFiltering, meanSmoothing, meanINPV))
                plt.xlabel('Hz')
            else:
                #plt.title(estimationClass + ' likelihood histogram; avg(F,S) = (%02.0f%%,%02.0f%%,%02.0f%%)' % (meanFiltering * 100, meanSmoothing * 100, meanINPV * 100))
                plt.xlabel('likelihood')
        else:
            if estimationClass == 'pitch':
                #plt.title(estimationClass + ' std histogram; avg(F,S) = (%02.0f,%02.0f) Hz' % (meanFiltering, meanSmoothing))
                plt.xlabel('Hz')
            else:
                #plt.title(estimationClass + ' likelihood histogram; avg(F,S) = (%02.0f%%,%02.0f%%)' % (meanFiltering * 100, meanSmoothing * 100))
                plt.xlabel('likelihood')
    plt.savefig(path2fig)
    print('fig saved')
    plt.show()


def calcStdOfMixOf2(means, covs, weights):
    return np.sqrt(weights[0]*covs[0] + weights[1]*covs[1] + weights[0]*np.power(means[0], 2) + weights[1]*np.power(means[1], 2) - np.power(weights[0]*means[0] + weights[1]*means[1], 2))

def limitFeatures(inputFeatures, chosenFeatures=[]):
    if chosenFeatures == []:
        y = inputFeatures
        raise Exception('I did not want to end up here')
    else:
        y = inputFeatures[:, chosenFeatures]
    return y
'''
def fx(x, snr=5):
    sig_dbm = 10*np.log10(np.power(np.percentile(np.abs(x), 80), 2)) + 30
    noise_dbm = sig_dbm - snr
    noise_std = np.sqrt(np.power(10, (noise_dbm - 30)/10))
    return x + noise_std*np.random.randn(x.size)
'''

def wordFeatureHistograms(path2WordFeatures):
    wordDatasetsFeatures = pickle.load(open(path2WordFeatures, "rb"))
    featureEntryIdx = 0
    nWords = len(wordDatasetsFeatures['train'][featureEntryIdx])
    wordIdx = 0
    nFeatures = np.min([wordDatasetsFeatures['train'][featureEntryIdx][0].shape[1], 13])
    n_bins = 100
    for featureIdx in range(nFeatures):
        plt.figure()
        for wordIdx in range(nWords):
            specificWordFeature = wordDatasetsFeatures['train'][featureEntryIdx][wordIdx][:, featureIdx]
            n, bins, patches = plt.hist(specificWordFeature, n_bins, density=True, histtype='step', cumulative=False, label='word %d' % wordIdx)
        plt.legend()
        plt.title('feature %d' % featureIdx)
        plt.show()

def meanFilteringSmoothingCalc(sentencesEstimationResults, maleIdx, femaleIdx, classCategories):
    # convert model first-word estimations to np-arrays:
    nSentences = len(sentencesEstimationResults)
    #classCategories = ['word']#, 'gender', 'speaker']
    collectedFirstWordSentenceResults = dict()
    for estimationClass in classCategories:
        collectedFirstWordSentenceResults[estimationClass] = dict()
        firstDigit_filtering, firstDigit_smoothing, firstDigit_raw = np.zeros(nSentences), np.zeros(nSentences), np.zeros(nSentences)
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
    meanFiltering = collectedFirstWordSentenceResults[estimationClass]['filtering'].mean()
    meanSmoothing = collectedFirstWordSentenceResults[estimationClass]['smoothing'].mean()
    return meanFiltering, meanSmoothing

def plotPitchHistogramPerSentence(sentencesDatasetsPitch, sentencesEstimationResults):
    n_bins = 20
    nGaussianComponents = sentencesEstimationResults[0]['results']['pitch']['smoothing']['weights'].size
    for sentenceIdx, sentence in enumerate(sentencesDatasetsPitch):
        plt.figure()
        for wordIdx, word in enumerate(sentence):
            if wordIdx == 0:
                speakerNo = int(word)
                continue
            pitchIndexes = word[1].nonzero()
            pitchValues = word[1][pitchIndexes]
            if wordIdx == 1:
                allPitchValues = pitchValues
            else:
                allPitchValues = np.hstack((allPitchValues, pitchValues))
        pitchMeans, pitchVars, pitchWeights = sentencesEstimationResults[sentenceIdx]['results']['pitch']['smoothing']['means'], sentencesEstimationResults[sentenceIdx]['results']['pitch']['smoothing']['covs'], sentencesEstimationResults[sentenceIdx]['results']['pitch']['smoothing']['weights']
        n, bins, patches = plt.hist(allPitchValues, n_bins, density=True, histtype='step', cumulative=False, label='sentence %d; speaker %d:' % (sentenceIdx, speakerNo))
        x = np.linspace(allPitchValues.min(), allPitchValues.max(), 200)
        allComponents = np.zeros(x.shape)
        for componentIdx in range(nGaussianComponents):
            mu, sigma, weight = pitchMeans[componentIdx], np.sqrt(pitchVars[componentIdx]), pitchWeights[componentIdx]
            singleComponent = weight * stats.norm.pdf(x, mu, sigma)
            plt.plot(x, singleComponent, label='component %d' % componentIdx)
            allComponents += singleComponent
        plt.plot(x, allComponents, label='mixture')


        plt.xlabel('Pitch [Hz]')
        plt.legend()
        #plt.xlim(50, 400)
        plt.show()

def createSentencesDataset(metadata, path2SentencesResults, path2SentencesMetadata, path2SentencesFeatures, path2SentencesAudio, path2SentencesPitch, path2WordModels, path2SpeakerModels, path2GenderModels, chosenFeatures, nSentences=10000, whichSet='train'):
    if os.path.isfile(path2SentencesResults):
        sentencesEstimationResults = pickle.load(open(path2SentencesResults, "rb"))
        sentencesMetadataTrain, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
    else:
        # create sentences dataset:
        if os.path.isfile(path2SentencesMetadata):
            sentencesMetadata, priorStates, transitionMat = pickle.load(open(path2SentencesMetadata, "rb"))
            # sentencesDatasetsAudio = pickle.load(open(path2SentencesAudio, "rb"))
            # sd.play(sentencesDatasetsAudio[0][1][1],fs)
            # sd.stop()
        else:
            sentencesMetadata, priorStates, transitionMat = createSentencesMetadata(metadata, path2SentencesMetadata, nSentences=nSentences, whichSet=whichSet)

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
            sentencesEstimationResults = createSentencesEstimationResults(sentencesDatasetsFeatures, sentencesDatasetsPitch, metadata, path2SentencesResults, path2WordModels, path2SpeakerModels, path2GenderModels, transitionMat, priorStates, trainOnLessFeatures=True, enableMahalanobisCala=False, chosenFeatures=chosenFeatures)
    return sentencesEstimationResults

def sampleFromSmoothing(sentencesEstimationResults, enableSimpleClassification=False):
    sentencesEstimationResults_sampled = np.zeros((len(sentencesEstimationResults), 4)) # saving as matrix. every row has [gender, speaker, word, pitch]
    for sentenceIdx, sentence in enumerate(sentencesEstimationResults):
        if enableSimpleClassification:
            sentenceResults = sentence['groundTruth']
            sentenceResults['pitch'] = sentence['results']['pitch']
            if sentenceResults['SpeakerGender'] == 'male':
                sampledGender = 1
            else:
                sampledGender = 0
            sampledSpeaker = int(sentenceResults['SpeakerNo'])
            sampledWord = sentenceResults['Digits'][0]
        else:
            sentenceResults = sentence['results']
            # extract multinomial probs:
            genderProbs, speakerProbs, wordProbs = sentenceResults['gender']['smoothing'][0], sentenceResults['speaker']['smoothing'][0], sentenceResults['word']['smoothing'][0]
            # sample:
            sampledGender, sampledSpeaker, sampledWord = int(np.argwhere(np.random.multinomial(1, pvals=genderProbs))), int(np.argwhere(np.random.multinomial(1, pvals=speakerProbs))), int(np.argwhere(np.random.multinomial(1, pvals=wordProbs)))
        sampledPitchMixture = int(np.argwhere(np.random.multinomial(1, pvals=sentenceResults['pitch']['smoothing']['weights'])))
        pitchMean, pitchStd = sentenceResults['pitch']['smoothing']['means'][sampledPitchMixture], np.sqrt(sentenceResults['pitch']['smoothing']['covs'][sampledPitchMixture])
        sampledPitch = np.random.normal(loc=pitchMean, scale=pitchStd)
        sentencesEstimationResults_sampled[sentenceIdx] = np.array([sampledGender, sampledSpeaker, sampledWord, sampledPitch])
    return sentencesEstimationResults_sampled

def generateAudioMatrix(sentencesDatasetsAudio):
    maxSentenceLengh = 0
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudio):
        if len(sentence[1][1]) > maxSentenceLengh: maxSentenceLengh = len(sentence[1][1])
    sentenceAudioMat = np.zeros((len(sentencesDatasetsAudio), maxSentenceLengh))
    for sentenceIdx, sentence in enumerate(sentencesDatasetsAudio):
        wav = sentence[1][1]
        zeroPaddedWav = np.concatenate((np.zeros(maxSentenceLengh - len(wav)), wav))
        sentenceAudioMat[sentenceIdx] = zeroPaddedWav
    return sentenceAudioMat

class VAE(nn.Module):
    def __init__(self, measDim, lstmHiddenSize, lstmNumLayers, nDrawsFromSingleEncoderOutput, zDim):
        super(VAE, self).__init__()

        self.nDrawsFromSingleEncoderOutput = nDrawsFromSingleEncoderOutput
        self.measDim = measDim
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
        return genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, mu, logvar, z


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

def loss_function_classification_prob(wordProbs, sampledWord):
    nDecoders = 1
    sampledWord = sampledWord.unsqueeze(1).repeat(nDecoders, 1).reshape(-1)
    wordNLL = F.cross_entropy(wordProbs, sampledWord, reduction='none')
    return wordNLL

def loss_function(mu, logvar, genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, sampledGender, sampledSpeaker, sampledWord, sampledPitch, nDecoders):
    batchSize = sampledWord.numel()
    sampledWord, sampledGender, sampledSpeaker, sampledPitch = sampledWord.unsqueeze(1).repeat(nDecoders, 1).reshape(-1), sampledGender.unsqueeze(1).repeat(nDecoders, 1).reshape(-1), sampledSpeaker.unsqueeze(1).repeat(nDecoders, 1).reshape(-1), sampledPitch.unsqueeze(1).repeat(nDecoders, 1).reshape(-1)

    genderNLL = F.cross_entropy(genderProbs, sampledGender, reduction='none')
    speakerNLL = F.cross_entropy(speakerProbs, sampledSpeaker, reduction='none')
    wordNLL = F.cross_entropy(wordProbs, sampledWord, reduction='none')

    pitchMean = torch.squeeze(pitchMean)
    pitchLogVar = torch.squeeze(pitchLogVar)

    pitchNLL = -torch.mul(pitchLogVar, 0.5) - torch.mul(torch.pow(torch.div(sampledPitch - pitchMean, torch.exp(torch.mul(pitchLogVar, 0.5))), 2), 0.5)
    '''
    t = time.time()
    pitchNLL = [torch.distributions.normal.Normal(pitchMean[pitchIdx], pitchLogVar[pitchIdx].mul(0.5).exp_()).log_prob(sampledPitch[pitchIdx]) for pitchIdx in range(pitchMean.shape[0])]
    print('loss function - pitch for duration: ', time.time() - t, ' sec')
    '''
    '''
    t = time.time()
    pitchNLL = torch.zeros(pitchMean.shape[0]).cuda()
    print('loss function - pitch cuda upload duration: ', time.time() - t, ' sec')
    
    t = time.time()
    for pitchIdx in range(pitchMean.shape[0]):
        pitchNLL[pitchIdx] = -torch.distributions.normal.Normal(pitchMean[pitchIdx], pitchLogVar[pitchIdx].mul(0.5).exp_()).log_prob(sampledPitch[pitchIdx])
    print('loss function - pitch for duration: ', time.time() - t, ' sec')
    '''

    #totalNLL_max = (genderNLL+speakerNLL+wordNLL+pitchNLL).reshape(-1, batchSize).max(dim=0)[0].sum()  # each column has the nDecoders different outputs from the same encoder's output, then max is performed
    totalNLL_max = (wordNLL).reshape(-1, batchSize).max(dim=0)[0].sum()

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return totalNLL_max# + KLD

def getProbabilitiesLUT(genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, nDecoders):
    batchSize = int(genderProbs.shape[0]/nDecoders)
    n_clusters = 3
    probabilitiesLUT = list()
    # the nDecoders samples for input no. i to the encoder is at genderProbs[:, i, :] :
    genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar = genderProbs.reshape(nDecoders, batchSize, -1), speakerProbs.reshape(nDecoders, batchSize, -1), wordProbs.reshape(nDecoders, batchSize, -1), pitchMean.reshape(nDecoders, batchSize, -1), pitchLogVar.reshape(nDecoders, batchSize, -1)
    for batchIdx in range(batchSize):
        singleInputGenderProb, singleInputSpeakerProb, singleInputWordProb, singleInputPitchMean, singleInputPitchStd = genderProbs[:, batchIdx, :], speakerProbs[:, batchIdx, :], wordProbs[:, batchIdx, :], pitchMean[:, batchIdx, :], np.exp(np.multiply(pitchLogVar[:, batchIdx, :], 0.5))
        sampledGender, sampledSpeaker, sampledWord, sampledPitch = np.zeros((nDecoders, 1)), np.zeros((nDecoders, 1)), np.zeros((nDecoders, 1)), np.zeros((nDecoders, 1))
        for decoderIdx in range(nDecoders):
            pvals = singleInputGenderProb[decoderIdx] / singleInputGenderProb[decoderIdx].sum()
            sampledGender[decoderIdx] = int(np.argwhere(np.random.multinomial(1, pvals=pvals)))
            pvals = singleInputSpeakerProb[decoderIdx] / singleInputSpeakerProb[decoderIdx].sum()
            sampledSpeaker[decoderIdx] = int(np.argwhere(np.random.multinomial(1, pvals=pvals)))
            pvals = singleInputWordProb[decoderIdx] / singleInputWordProb[decoderIdx].sum()
            sampledWord[decoderIdx] = int(np.argwhere(np.random.multinomial(1, pvals=pvals)))
            sampledPitch[decoderIdx] = np.random.normal(loc=singleInputPitchMean[decoderIdx], scale=singleInputPitchStd[decoderIdx])
        X = np.concatenate((sampledGender, sampledSpeaker, sampledWord, sampledPitch), axis=1)
        if X.shape[0] == 1: # single decoder
            clustersWeights = np.ones(1)
            clustersRepresentatives = X
        else:
            # normalize X:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # cluster X with KMeans:
            kmeans = KMeans(n_clusters=n_clusters).fit(X_scaled)
            clustersWeights = [(kmeans.labels_ == clusterIdx).sum()/nDecoders for clusterIdx in range(n_clusters)]
            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_scaled)
            clustersRepresentatives = X[closest]
        probabilitiesLUT.append([clustersWeights, clustersRepresentatives])
    return probabilitiesLUT

def trainFunc(sentencesAudioInputMatrix, sentencesEstimationResults_sampled, sentencesEstimationPitchResults_sampled, model, optimizer, epoch, validateOnly=False, enableSimpleClassification=False):
    if validateOnly:
        model.eval()
    else:
        model.train()

    total_loss = 0

    batchSize = 200
    nSentences = sentencesAudioInputMatrix.shape[0]
    nBatches = int(nSentences/batchSize)
    nDecoders = model.nDrawsFromSingleEncoderOutput

    inputSentenceIndexes = torch.randperm(nSentences)
    sentencesAudioInputMatrix, sentencesEstimationResults_sampled, sentencesEstimationPitchResults_sampled = sentencesAudioInputMatrix[inputSentenceIndexes], sentencesEstimationResults_sampled[inputSentenceIndexes], sentencesEstimationPitchResults_sampled[inputSentenceIndexes]
    probabilitiesLUT = list()
    for batchIdx in range(nBatches):
        if batchIdx % 10 == 0: print('epoch %d: starting batch %d out of %d' % (epoch, batchIdx, nBatches))
        data = sentencesAudioInputMatrix[batchIdx * batchSize:(batchIdx + 1) * batchSize]
        # crop beginning to have integer size of model.measDim and then reshape to have model.measDim features:
        nSamples2Crop = data.shape[1] - int(data.shape[1] / model.measDim) * model.measDim
        data = (data[:, nSamples2Crop:]).reshape(data.shape[0], -1, model.measDim).transpose(1, 0).float()

        labels, pitchLabels = sentencesEstimationResults_sampled[batchIdx * batchSize:(batchIdx + 1) * batchSize].long(), sentencesEstimationPitchResults_sampled[batchIdx * batchSize:(batchIdx + 1) * batchSize].float()
        sampledGender, sampledSpeaker, sampledWord, sampledPitch = labels[:, 0], labels[:, 1], labels[:, 2], pitchLabels[:, 0]

        if not validateOnly: optimizer.zero_grad()
        genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, mu, logvar, z = model(data)
        # t = time.time()

        if enableSimpleClassification:
            loss = loss_function_classification_prob(z, sampledWord)
        else:
            loss = loss_function(mu, logvar, genderProbs, speakerProbs, wordProbs, pitchMean, pitchLogVar, sampledGender, sampledSpeaker, sampledWord, sampledPitch, model.nDrawsFromSingleEncoderOutput)

        # print('loss function duration: ', 1000*(time.time()-t), ' ms')
        if not validateOnly: loss.backward()
        total_loss += loss.item() / batchSize
        if not validateOnly: optimizer.step()

        if validateOnly: probabilitiesLUT += getProbabilitiesLUT(F.softmax(genderProbs, dim=1).cpu().detach().numpy(), F.softmax(speakerProbs, dim=1).cpu().detach().numpy(), F.softmax(wordProbs, dim=1).cpu().detach().numpy(), pitchMean.cpu().detach().numpy(), pitchLogVar.cpu().detach().numpy(), nDecoders)

    return total_loss / nBatches, probabilitiesLUT

