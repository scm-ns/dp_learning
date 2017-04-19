# Rewrite/copy of Aymeric Damien


import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

Xtre , Ytr = mnist.train.next_batch(5000)
Xte , Yte = mnist.test.next_batch(200)


distance = tf.reduce_sum(tf.add(xtr , tf.negative(xte))) , reduction_indices = 1)
pred = tf.arg_min(distance , 0)


accuracy = 0.

init = tf.global_variables_initializer()


with tf.Session() as sess:



