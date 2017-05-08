import numpy as np
import tensorflow as tf

net_pretrained = np.load("bvlc-alexnet.npy" , ecoding="latin1").item()


def conv(input , kernel , biases , kernel_h , kernel_w , num_channels , stride_h , stride_w , padding="VALID" , group=1):
    
    c_i = input.get_shape()[-1]
    assert c_i % group == 0 
    assert num_channels % group == 0
    convolve = lambda i, k : tf.nn.conv2d(i , k , [1, stride_h , stride_w , 1], padding=padding)

    if group == 1:
        conv = convolve(input , kernel)
    else:
        input_groups = tf.split(3 , group , input) # how to divide the input into groups with
        kernel_groups = tf.split(3 , group , kernel)
        output_groups = [convolve(i,k) for i , k in zip(input_groups , kernel_groups)]
        conv = tf.concat(3 , output_groups)
    # reshape the tensor, may be for feeding or reading from caffe ??
    return tf.reshape(tf.nn.bias_add(conv , biases) , [-1] + conv.get_shape().as_list()[1:])


def AlexNet(features , features_extract = False):
    # Builds model and loads pretrained model
    kernel_size = 11
    stride = 4
    channels = 96
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Varaible(net_data["conv1"][1])
    conv1_in = conv(features, conv1W , conv1b , kernel_size , kernel_size ,channels , stride , stride , padding="SAME" , group = 1)
    conv1 = tf.nn.rele(conv1_in)

    # normalization ?? 
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    tf.nn.local_response_normalization(conv
    lrn1 = tf.nn.local_response_normalization(conv1 , depth_radius = radius , alpha = alpha , beta=beta , bias = bias)

    # maxpool1 
    kernel = 3
    stride = 2
    padding= "VALID"
    maxpool1 = tf.nn.max_pool(lrn1 , ksize=[1 , kernel , kernel , 1] , stride=[1 , stride , stride , 1] , padding =padding)

    # conv2
    kernel = 5
    channels = 256
    stride = 1
    group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Varaible(net_data["conv2"][1])
    conv2_in = conv(maxpool1 , conv2W , conv2b , kernel , kernel, channels , stride , stride , padding="SAME", group =group)

    #lrn2
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2 , depth_radius = radius , alpha = alpha , beta = beta , bias = bias)

    # maxpool2
    kernel = 3
    stride = 2
    padding ="VALID"
    maxpool2 = tf.nn.max_pool(lrn2 , ksize=[1 , kernel , kernel , 1], strides =[1 , stride , stride , 1], padding = padding)








