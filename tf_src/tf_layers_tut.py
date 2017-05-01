from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensroflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features , [-1 , 28 , 28 , 1])

    conv1 = tf.layers.conv2d(inputs = input_layer , filters = 32 , kernel_size = [5 , 5] , padding ="same" , 
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1 , pool_size = [2 , 2] , strides = 2)

    # Convolutional layer #2 
    conv2 = tf.layers.conv2d(inputs = pool1 , filters = 64 , kernel_size = [5,5] , padding="same" , 
            activation = tf.nn.relu)
    

    pool2 =  tf.layers.max_pooling2d(inputs = conv2 , pool_size=[2,2] , strides = 2)

    # Dense layers 
    pool2_flat = tf.reshape(pool2 , [-1 , 7 * 7 * 64])  # Flatten the layer to feed it into the dense layer
     
    dense = tf.layers.dense(inputs =pool2_flat , units = 1024 , activation = tf.nn.relu)

    dropout = tf.layers.droput(inputs = dense , rate = 0.4 , training=mode == learn.ModeKeys.Train)

    logits = tf.layers.dense(inputs= dropout , units = 10)
    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32) , depth = 10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels , logits = logits)

    if mode  == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(uloss = goss , global_step =tf.contrib.framework.get_global_step(),learning_rate = 0.001 , optimizer="SGD")


    predictions = {
            "classes" : tf.argmax(input = logits,axis = 1),
            "probabilities" : tf.nn.softmax(logits , name="softmax_tensor")
    }


    return model_fn_lib.ModelFnOps(mode = mode , predictions = predictions , loss = loss , train_op = train_op)




if __name__ == "__main__" :
    tf.app.run()


