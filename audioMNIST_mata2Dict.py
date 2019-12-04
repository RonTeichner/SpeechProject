import os
import numpy as np
from scipy.io import wavfile
from speakerfeatures import *
import matplotlib.pyplot as plt
import random
import pickle
from hmmlearn.hmm import GMMHMM as GMMHMM
from sklearn.mixture import GaussianMixture



def createAudioMNIST_metadataDict():
    path2AudioMNIST = '/media/ront/Files/Projects/AudioMNIST/data/'
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

def createSpeakerWavs_Features(metadata, fs, path2SpeakerAudio, path2SpeakerFeatures, type):
    speakerDatasetsAudio = dict()
    speakerDatasetsFeatures = dict()
    for dataset in ['train', 'validate', 'test']:
        if type == 'speakers':
            speakerWavs = metadata.get_speaker_set(dataset)
        elif type == 'genders':
            speakerWavs = metadata.get_gender_set(dataset)

        speakerAudio = list(range(len(speakerWavs)))
        speakerAudioLengths = list(range(len(speakerWavs)))
        speakerFeatures = list(range(len(speakerWavs)))
        speakerFeaturesLengths = list(range(len(speakerWavs)))
        for speakerIdx, singleSpeakerList in enumerate(speakerWavs):
            for wavIdx, wav in enumerate(singleSpeakerList):
                if wavIdx%100 == 0:
                    print('dataset ',dataset,' starting speaker %d feature extraction; wavIdx %d out of %d' % (speakerIdx, wavIdx, len(singleSpeakerList)))
                extractedFeatures = np.float32(extract_features(wav, fs))
                wav = np.expand_dims(wav, axis=1)
                if wavIdx == 0:
                    speakerAudio[speakerIdx] = wav
                    speakerAudioLengths[speakerIdx] = list()
                    speakerFeatures[speakerIdx] = extractedFeatures
                    speakerFeaturesLengths[speakerIdx] = list()
                else:
                    speakerAudio[speakerIdx] = np.vstack((speakerAudio[speakerIdx], wav))
                    speakerFeatures[speakerIdx] = np.vstack((speakerFeatures[speakerIdx], extractedFeatures))
                speakerAudioLengths[speakerIdx].append(wav.shape[0])
                speakerFeaturesLengths[speakerIdx].append(extractedFeatures.shape[0])
        speakerAudioList = [speakerAudio, speakerAudioLengths]
        speakerFeaturesList = [speakerFeatures, speakerFeaturesLengths]
        speakerDatasetsAudio[dataset] = speakerAudioList
        speakerDatasetsFeatures[dataset] = speakerFeaturesList
    pickle.dump(speakerDatasetsAudio, open(path2SpeakerAudio, "wb"))
    pickle.dump(speakerDatasetsFeatures, open(path2SpeakerFeatures, "wb"))
    return speakerDatasetsFeatures


def speakerClassificationTrain(speakerDatasetsFeatures, path2SpeakerModels):
    nSpeakers = len(speakerDatasetsFeatures['train'][0])
    covariance_type = 'diag'
    nTrainIters = 5
    max_nCorrect = -np.inf
    for trainIter in range(nTrainIters):
        nHmmStates = 1
        speakerModels = [GMMHMM(n_components=nHmmStates, n_mix=3, n_iter=200, covariance_type=covariance_type) for speakerIdx in range(nSpeakers)]
        for speakerIdx in range(nSpeakers):
            lengthsVec = np.asarray(speakerDatasetsFeatures['train'][1][speakerIdx])
            if nHmmStates == 1:
                # empirically, train results are better with all the gender signals collapsed to a single signal
                lengthsVec = np.expand_dims(lengthsVec.sum(), axis=0)
            speakerModels[speakerIdx].fit(speakerDatasetsFeatures['train'][0][speakerIdx], lengths=lengthsVec)

        # validation:
        datasetKey = 'validate'
        nCorrect = 0
        nExamples = 0
        genderResults = list()
        for speakerIdx in range(nSpeakers):
            nValExamples = np.asarray(speakerDatasetsFeatures[datasetKey][1][speakerIdx]).shape[0]
            speakerResult = np.zeros((nValExamples, nSpeakers))
            startIdx = 0
            for singleValIdx in range(nValExamples):
                singleValLength = speakerDatasetsFeatures[datasetKey][1][speakerIdx][singleValIdx]
                stopIdx = startIdx + singleValLength
                wavFeatures = speakerDatasetsFeatures[datasetKey][0][speakerIdx][startIdx:stopIdx]
                for modelIdx in range(nSpeakers):
                    speakerResult[singleValIdx, modelIdx] = speakerModels[modelIdx].score(wavFeatures)
                startIdx += singleValLength
            predictedSpeaker = np.argmax(speakerResult, axis=1)
            nCorrectSpeaker = (predictedSpeaker == speakerIdx).sum()
            nCorrect += nCorrectSpeaker
            nExamples += nValExamples
            genderResults.append(speakerResult) # for future use
            print('hmmgmm: groupIdx: %d: %d correct out of %d <=> %02.0f%%' % (speakerIdx, nCorrectSpeaker, nValExamples, nCorrectSpeaker/nValExamples*100))
        if nCorrect > max_nCorrect:
            max_nCorrect = nCorrect
            speakerModels2Save = speakerModels
        print('best performance on validation set: %d correct out of %d <=> %02.0f%%' % (nCorrect, nExamples, nCorrect / nExamples * 100))

    pickle.dump(speakerModels2Save, open(path2SpeakerModels, "wb"))