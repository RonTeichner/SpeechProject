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
from pysndfx import AudioEffectsChain
from copy import deepcopy
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import scipy.stats as stats

n_bins = 20

mu = 170
sigma = 300
weight = 1
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
n = mu + sigma*np.random.randn(200, 1)
GaussianMixModel = GaussianMixture(n_components=1, covariance_type='diag', reg_covar=1e-1).fit(n)
plt.hist(n, n_bins, density=True, histtype='step', cumulative=False, label='hist')
plt.plot(x, weight * stats.norm.pdf(x, mu, sigma), label='gaussian')
plt.plot(x, weight * stats.norm.pdf(x, GaussianMixModel.means_[0][0], np.sqrt(GaussianMixModel.covariances_[0][0])), label='fit')
plt.legend()
# plt.xlim(100, 200)
plt.show()
