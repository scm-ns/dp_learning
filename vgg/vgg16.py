"""
    Based off of : http://www.cs.toronto.edu/~frossard/post/vgg16/ 
"""


import tensorflow as tf
import numpy as np

class vgg16:
    def __init__(imgs):
        # remove imagenet image means
        mean = tf.constant([123.68 , 116.779 , 103.939] , dtype= tf.float32 , shape=[1 , 1 , 1 , 3] , name = "img_mean")
        net_in = imgs - mean
   
        conv_1_1_out;
        with tf.name_scope("conv_1_1") as scope: 
            in_size = 3;
            out_size = 64;
            weights = tf.Variable(tf.truncatedNormal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(net_in , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_1_1_out = tf.nn.relu(out)

 
        conv_1_2_out;
        with tf.name_scope("conv_1_2") as scope: 
            in_size = 64;
            out_size = 64
            weights = tf.Variable(tf.truncatedNormal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_1_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_1_2_out = tf.nn.relu(out)

        pool_1 = tf.nn.max_pool(conv_1_2_out , ksize = [1 ,2 , 2 , 1] , strides = [1 ,2 ,2 , 1], padding="SAME" )
        

        conv_2_1_out;
        with tf.name_scope("conv_2_1") as scope: 
            in_size = 64
            out_size = 128
            weights = tf.Variable(tf.truncatedNormal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(pool_1, weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_2_1_out = tf.nn.relu(out)

        conv_2_2_out;
        with tf.name_scope("conv_2_2") as scope: 
            in_size = 128
            out_size = 128
            weights = tf.Variable(tf.truncatedNormal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_2_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_2_1_out = tf.nn.relu(out)

        pool2 = tf.nn.max_pool(conv_2_2_out , ksize = [1 , 2 ,2 , 1] , strides= [1 , 2 ,2 ,1 ], padding = "SAME" )












