from __future__ import division , print_function

import os , json
from glob import glob
import numpy as np
from scipy import misc , ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normaization import BatchNormalization
from keras.util.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense , Dropout , Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D , ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD , RMSprop , Adam
from keras.preprocessing import image


vgg_mean = np.array([123.68 , 116.779 , 103.939] , dtype=np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb -. bgr

class vgg18_bn():



    def __init__(self , size= (224 , 244), include_top= True):
        self.FILE_PATH = "http://www.platform.ai/models/"
        self.create(size , include_top)

        self.get_classes()


    def get_classes(self):
        fname = "imagenet_class_index.json"
        fpath = get_file(fname , self.FILE_PATH + fname , cache_subdit = "models")
        with 








