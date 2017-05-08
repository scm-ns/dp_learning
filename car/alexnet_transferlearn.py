import numpy as np
import tensorflow as tf

train_x =  np.zeros((1, 277 , 227 , 3),dtype=np.float)
train_y = np.zeros((1,1000),dtype=np.int)
xdim = train_x.shape[1:]
ydim = train_y.shape[1:]

from alexnet import AlexNet
from caffe_classes import class_names

x = tf.placeholder(tf.float32 ,(None,) + xdim)
resized = tf.image.resize_images(x , (227,227))



prob = 


