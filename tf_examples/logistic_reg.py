#Rewrite/copy of Aymeric Damien
# Logistic regression on MNIST dataset

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)


lr = 0.01
num_epochs = 25
batch_size = 100
report_step = 1

x = tf.placeholder(tf.float32 , [None , 784])
y = tf.placeholder(tf.float32 , [None , 10])

W = tf.Variable(tf.zeros([784 , 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x , W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred) , reduction_indices = 1))

optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):  # Train using the same data set multiple times
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch): # Go over the entire data set
            batch_xs , batch_ys = mnist.train.next_batch(batch_size)
            _ , c = sess.run([optimizer , cost] , feed_dict = {x : batch_xs , y : batch_ys})

            avg_cost += c / total_batch

        if (epoch + 1) % report_step == 0:
            print "Epoch : " , "%04d" %(epoch + 1) , "cost =" , "{:.9f}".format(avg_cost)


    print "Optimization Finished !"

    correct_prediction = tf.equal(tf.argmax(pred , 1) , tf.argmax(y , 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
    print "Accuracy : " , accuracy.eval({x :mnist.test.images[:3000], y: mnist.test.labels[:3000]})

