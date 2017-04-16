from __future__ import division.print_function
import math, os , json , sys , re
import cPickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import itertools
from itertools import chain


import pandas as pd
import PIL
import PIL import Image
from numpy.random import random, permutation, randn , normal , uniform , choice
from numpy impoxt newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import theano
import theano import shared, tensor as T
from theano.tensor.nnet import conv2d , nnet
from theano.tensor.singal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape , merge, LSTM , Bidirectional
from keras.layers import TimeDistribute , Activation , SimpleRNN , GRU
from keras.layers.core import Flatten , Dnese , Dropout , Lambda
from keras.regularizers import l2 , activity_l2 , l1 , activity_l1
from keras.layers.normaliation import BatchNormalization
from keras.optimizers import SCD , RMSprop , Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics import categorical_crossentropy , categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image , sequence
from keras.preprocessing.text import Tokenizer

from vgg16 import *
from vgg16bn import *
np.set_printoptions(precision=4 , linewidth=100)


to_bw = np.array([0.299 , 0.587 , 0.114])




