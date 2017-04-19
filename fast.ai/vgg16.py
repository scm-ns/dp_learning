from __future__ import divisoin , print_function

import os , json
from glob import glob
import numpy as np
from scipy import misc , ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten , Dense , Dropout , Lambda
from keras.layers.convolutional import Convolution2D , MaxPooling2D , ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD , RMSprop , Adam
from keras.preprocessing import image


vgg_mean = np.array([123.68 , 116.779 , 103.939] , dtype = np.float32).reshape((3,1,1))

def vgg_preprocess(x):
    x = x - vgg_mean
    return x[: , ::-1]


class vgg_16():


    def __init__(self):
        self.FILE_PATH = "http://www.platform.ai/models/"
        self.create()
        self.get_classes()


    def get_classes(self):
        fname = "imagenet_class_index.json"
        fpath = get_file(fname , self.FILE_PATH + fname , cache_subdir = "models")
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self , imgs , details = False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds , axis = 1)
        preds = [all_preds[i , idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds) , idxs , classes

    def conv_block(self , layers , filters):
        model = self.model
        for i in range(layers):
            model.app(ZeroPadding2D((1,1))
            model.add(convolutional2D(filters , 3 , 3 , activation = "relu"))
        model.add(MaxPooling2D((2,2) , strides=(2,2)))


    def fc_block(self):
        model = self.model
        model.add(Dense(4096 , activation = "relu"))
        model.add(Dropout(0.5))


    def create(self):
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess , input_shape = ( 3 , 224 , 224) , output_shape=(3,244,244)))


        self.conv_block(2,64)
        self.conv_block(2,128)
        self.conv_block(3,256)
        self.conv_block(3,512)
        self.conv_block(3,512)


        model.add(Flatten())
        self.fc_block()
        self.fc_block()
        model.add(Dense(1000 , activation = "softmax"))

        fname = "vgg16.h5"
        model.load_weights(get_file(fname , self.FILE_PATH + fname , cache_subdir = "models"))



    def get_batches(self , path , gen = image.ImageDataGenerator() , shuffle = True , batch_size = 8 , class_mode = "categorical"):
        return gen.flow_from_directory(path , target_size=(244,244) , 
            class_mode = class_mode , shuffle = shuffle , batch_size= batch_size)


    def ft(self, nums):
        model = self.model
        model.pop() # remove the last layer
        for layers in model.layers:
            layer.trainable = False
        model.add(Dense(num , activation = "softmax"))
        self.compile()



    def fine_tune(self , batches):
        self.ft(batches.nb_class)
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self , lr = 0.001):
        self.model.compile(optimizer=Adam(lr = lr),
                    loss="categorical_crossentropy" , metrics=["accuracy"])

    def fit_data(self , trn , labels , val , val_labels , nb_epoch = 1 , batch_size = 64):
        self.model.fit(trn , labels , nb_epoch = nb_epoch , validation_data=(val , val_labels),
                batch_size = batch_size)


    def fit(self , batches , val_batches , nb_epoch = 1):
        self.model.fit_generator(batches , samples_per_epoch = batches.nb_smaple , nb_epoch = nb_epoch , validation_data = val_batches , nb_val_samples = val_batches.nb_sample)



    def test(self , path , batch_size = 8):
        test_batches = self.get_batches(path , suffle = False , batch_size = batch_size , class_mode = None)
        return test_batches , self.model.predict_generator(test_batches , test_batches.nb_sample)





















