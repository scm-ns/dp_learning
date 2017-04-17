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




