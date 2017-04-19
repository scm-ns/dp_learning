# Copy of lesson 1 Practical Deep learning fastai



path = "data/dogscats/"

from __future__ import division, print_function

import os , json 
from glob import glob
import numpy as np
np.set_printoptions(precision = 4 , linewidth = 100)
from matplotlib import pyplot as plt

import utils
from utils import plots

batch_size = 64


import vgg16
from vgg16 import Vgg16



"""
    The directory must be named the category of the object when using vgg16
    We are using a pretrained network to distinguish between 1000 different objects
"""

vgg = Vgg16()
img , labels = next(batches)
plots(imgs , titles=labels)
vgg.predict(imgs , True)
vgg.classes[:4]

batches = vgg.get_batches(path + "train" , batch_size = batch_size)

vgg.finetune(batches) # This modifies the model so that it can be trained on a new set of classes
# What it is essentially doing is removing the last softmax layers and replacing it with another softmax layer


val_batches = vgg.get_batches(path + "valid" , batch_size = batch_size * 2)
vgg.fit(batches , val_batches , nb_epoch = 1)

## Set up the network from scratch

from numpy.random import random , permutation
from scipy import misc , ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential , Model
from keras.layers.core import Flatten , Dense , Dropout , Lambda
from keras.layers import Input
from keras.layers.convoutional import Convolution2D , MaxPooling2D , ZeroPadding2D
from keras.optimizers import SGD , RMSprop
from keras.preprocessing import image 

FILES_PATH = "http://www.platform.ai.models/"
CLASS_FILE = "imagenet_class_index.json"

fpath = get_file(CLASS_FILE , FILES_PATH + CLASS_FILE , cache_subdir = "models")
with open(fpath) as f :
    class_dict =  json.load(f)
classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

classes[:5]

def conv_block(layers, model , filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(filters , 3 , 3 , activation = "relu"))
    model.add(MaxPooling2D((2,2) , strides = (2,2)))


def fc_block(model):
    model.add(Dense(4096 , activation = "relu"))
    model.add(Dropout(0.5))

# Mean of each channel as by VGG reserachers. But it is not just for their data set ? 
# May be we since we are using their pretrained model, we need to preprocess our images 
# in the same way to get the same result. 


vgg_mean = np.array([123.68 , 116.779 , 103.939]).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean # remove the mean
    return x[: , ::-1] # convert from BGR to RGB


def VGG_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess , input_shape = ( 3 , 224 , 224)))

    conv_block(2 , model , 64)
    conv_block(2 , model , 128)
    conv_block(3 , model , 256)
    conv_block(3 , model , 512)
    conv_block(3 , model , 512)

    model.add(Flatten())
    fc_block(model)
    fc_block(model)
    model.add(Dense(1000 , activation ="softmax"))
    return model 

model = VGG_16()

fpath = get_file("vgg16.h5" , FILES_PATH+"vgg16.h5" , cache_subdir="models")
model.load_weights(fpath) # This will input all the model weights into the model scafold

# Now since we use a pretrained model, we donot have to train the network. 

batch_size = 4

def get_batches(dirname , gen = image.ImageDataGenerator() , shuffle = True, batch_size = batch_size , class_model ="categorical"):
    return gen.flow_from_directory(path + dirname, target_size = (224 , 224), class_model = class_model , shuffle = shuffle , batch_size = batch_size)


batches = get_batches("train" , batch_size = batch_size)
val_batches = get_batches("valid" , batch_size = batch_size)
imgs , labels = next(batches)

plots(imgs , titles = labels)

def pred_batch(imgs):
    preds = model.argmax(imgs) # This step does the predictions, given the images
    idxs = np.argmax(preds ,axis = 1)

    print("Shape : {}".format(preds.shape))
    print("First 5 classes : {}".format(classes[:5])
    print("First 5 probabilites  : {}\n".format(preds[0, :5]))
    print("Predictions prob/class: ")

    for i in range(len(idx)):
        idx = idxs[i]
        print ("  {:.4f}/{}".format(preds[i, idx] , classes[idx]))


pred_batch(imgs)

