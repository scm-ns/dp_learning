# Copy of fast.ai lessons

from __future__ import division, print_function
import os , json
from glob import glob
import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
np.set_printoptions(precision = 4 , linewidth = 100)
from matplotlib import pyplot as plt
#from utils import plots , get_batches , plot_confusion_matrix , get_data


from numpy.random import random , permutation 
from scipy import misc , ndimage
from scipy.ndimage.interpolation import zoom


import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Flatten , Dense , Dropout , Lambda
from keras.layers.convolutional import Convolution2D , MaxPooling2D , ZeroPadding2D
from keras.optimizers import SGD , RMSprop
from keras.preprocessing import image


x = random((30 , 2))
y = np.dot(x , [2. , 3.]) + 1. # Numpy uses column matrix as default

print(x[:5])

print(y[:5])

lm = Sequential([Dense( 1 , input_shape=(2,))])
lm.compile(optimizer = SGD(lr = 0.1) , loss = "mse")
lm.evaluate(x , y , verbose = 0)
lm.fit(x , y , nb_epoch =5 , batch_size = 1)
lm.evaluate(x , y , verbose = 0)

print(lm.get_weights())

path = "data/dogscats/"
model_path = path + "models/"
if not os.path.exists(model_path):
    os.mkdir(model_path)

batch_size = 100

from vgg16 import Vgg16
vgg = Vgg16()
model = vgg.model


val_batches = get_batches(path + "valid"  , shuffle = False , batch_size = 1)
batches = get_batches(path + "train" , shuffle = Flase , batch_size = 1)

import bcolz
def save_array(fname , arr): 
    c=bcolz.carray(arr , rootdir = fname , mode = "w");
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]


val_data = get_data(path + "valid")
trn_data = get_data(path + "train")


trn_data.shape

save_array(model_path + "train_data.bc" , trn_data)
save_array(model_path + "valid_data.bc" , val_data)


trn_data = load_array(model_path + "train_data.bc")
val_data = load_array(model_path + "valid_data.bc")


val_data.shape


def onehot(x): # Convert from column of classes into one hot encoding
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1 , 1)).todense()



val_classes = val_batches.classes
trn_classes = batches.classes
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)



trn_labels.shape


trn_classes[:4]


trn_labels[:4]


trn_features = model.predict(trn_data , batch_size = batch_size)
val_features = model.predict(val_data , batch_size = batch_size)


trn_features.shape


save_array(model_path + "train_lastlayer_features.bc" , trn_features)
save_array(model_path + "valid_lastlayer_features.bc" , val_features)

trn_features = load_array(model_path + "train_lastlayer_features.bc")
val_features = load_array(model_path + "valid_lastlayer_features.bc")


lm = Sequential([ Dense( 2 , activation = "softmax" , input_shape = (1000,)) ])
lm.compile(optimizer=RMSprop(lr=0.1) , loss="categorical_crossentropy" , metrics= ["accuracy"])


batch_size = 64


batch_size = 4


lm.fit(trn_features , trn_labels, nb_epoch = 3 , batch_size =batch_size, 
    validation_data =(val_features, val_labels))


lm.summary()

preds = lm.predict_classes(val_features , batch_size = batch_size)
probs = lm.predict_proba(val_features , batch_size = batch_size)[:,0]

print(preds[:8])

print(probs[:8])

filenames = val_batches.filenames

n_view = 4

def plots_idx(idx , titles = None):
    plots([ image.load_img(path + "valid/" + filenames[i]) for i in idx] , titles = titles)

# Correct labels :
correct = np.where(preds == val_labels[:,1])[0]
idx = premutation(correct)[:n_view]
plots_idx(idx , probs[idx])

incorrect = np.where(preds != val_labels[:,1])[0]
idx = permutation(incorrect)[:n_view]
plots_idx(idx , probs[idx])

correct_cats = np.where((preds == 0) & (preds == val_labels[:,1]))[0]
most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
plots_idx(correct_cats[most_correct_cats] , probs[correct_cats][most_correct_cats])

correct_dogs = np.where((predds == 1) & (preds == val_labels[:,1]))[0]
most_correct_docs = np.argsort(probs[correct_dogs])[:n_view]
plots_idx(correct_dogs[most_correct_dogs] , 1 - probs[correct_dogs][most_correct_dogs])


incorrect_cats = np.where((preds == 0) & (preds !=val_labels[:,1]))[0]
most_incorrect_cats = np.argsort(probs[incorrect_cats])[::-1][:n_view]
plots_idx(incorrect_cats[most_incorrect_cats] , probs[incorrect_cats][most_incorrect_cats])

incorrect_dogs = np.where((preds == 1) & (preds != val_labels[:,1]))[0]
most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[::-1][:n_view]
plots_idx(incorrect_dogs[most_incorrect_dogs] , probs[incorrect_dogs][most_incorrect_dogs])

most_uncertain = np.argsort(np.abs(probs - 0.5))
plots_idx(most_uncertain[:n_view] , probs[most_uncertain])

cm = confusion_matrix(val_classes , preds)


plot_confusion_matrix(cm , val_batches.class_indices)

#lm = Sequential([Dense( 2 , activation="softmax" , input_shape=(1000,))])

#model.add(Dense(4096 , activation = "relu"))

vgg.model.summary()

model.pop()
for layer in model.layers:
    layer.trainable = False











































































