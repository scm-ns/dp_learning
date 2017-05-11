import numpy as np
import tensorflow as tf
import time

from scipy.misc import imread
from scipy.misc import imresize

from alexnet import AlexNet
from caffe_classes import class_names

input_img_dim = (32 , 32 , 3)
custom_layer_layer_classes = 43

x = tf.placeholder(tf.float32 ,(None,) + input_img_dim)
resized = tf.image.resize_images(x , (227,227))

prob = alex_net(resized , custom_layer_layer_classes)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#im1 = (imread("poodle.png")[:,:,:3]).astype(np.float32)
im1 = (imread("construction.jpg")[:,:,:3]).astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = (imread("stop.jpg")[:,:,:3]).astype(np.float32)
im2 = im2 - np.mean(im2)

t = time.time()
# output is not a single images predicition, but concatenation of mulitpl images's prediction
output = sess.run(prob , feed_dict ={x:[im1 , im2]})


for input_im_index in range(output.shape[0]):
    # lines identifies the most probable class for each of images
    indexs = np.argsort(output)[input_im_index,:] # arrange in increasing order
    print("Image : " , input_im_index)
    for i in range(5): # print the top 5 classes, indexs , the most probable classes are towards the end
        print("%s: %.3f" % (indexs[-1-i], output[input_im_index , indexs[-1-i]])) 
    print()

print("Time :%.3f seconds" % (time.time() - t))




