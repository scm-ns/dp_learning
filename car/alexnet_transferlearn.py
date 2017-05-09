import numpy as np
import tensorflow as tf
import time

from scipy.misc import imread
from scipy.misc import imresize


from alexnet import AlexNet
from caffe_classes import class_names

input_img_dim = (32 , 32 , 3)

x = tf.placeholder(tf.float32 ,(None,) + input_img_dim)
resized = tf.image.resize_images(x , (227,227))

prob = AlexNet(resized)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#im1 = (imread("poodle.png")[:,:,:3]).astype(np.float32)
im1 = (imread("construction.jpg")[:,:,:3]).astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = (imread("stop.jpg")[:,:,:3]).astype(np.float32)
im2 = im2 - np.mean(im2)

t = time.time()
output = sess.run(prob , feed_dict ={x:[im1 , im2]})


for input_im_index in range(output.shape[0]):
    indexs = np.argsort(output)[input_im_index,:]
    print("Image : " , input_im_index)
    for i in range(5):
        print("%s: %.3f" % (class_names[indexs[-1-i]], output[input_im_index , indexs[-1-i]]))
    print()

print("Time :%.3f seconds" % (time.time() - t))




