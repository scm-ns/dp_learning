# Rewrite/copy of Aymeric Damien


import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

Xtr , Ytr = mnist.train.next_batch(5000)
Xte , Yte = mnist.test.next_batch(200)

xtr = tf.placeholder("float" , [None, 784])
xte = tf.placeholder("float" , [784])

distance = tf.reduce_sum(tf.add(xtr , tf.negative(xte))) , reduction_indices = 1)
pred = tf.arg_min(distance , 0)


accuracy = 0.

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):

        # HOw this works, it uses the training data to decide which data points in the training data set
        # the test data point is similar to. 
        # We choose the data point which is the most similar to the test data point.
        # Get its index and assume that the label of this index ie the class which it belongs to
        # is the same class that test data point belongs to. Too simplistic.

        nn_index = sess.run(pred , feed_dict = {xtr : Xtr ,xte : Xte[i , :]})

        print "Test" , i , "Predictions : " , np.argmax(Ytr[nn_index]) , "True class : " , np.argmax(Yte[i]_


        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print "Done!"
    print "Accuracy: " , accuracy





