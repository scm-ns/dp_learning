import numpy as np
import tensorflow as tf

net_pretrained = np.load("bvlc_alexnet.npy" , encoding="latin1").item()

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


def AlexNet(features , for_transfer_learning = False):
    # Builds model and loads pretrained model
    kernel_size = 11
    stride = 4
    channels = 96
    conv1W = tf.Variable(net_pretrained["conv1"]["weights"])
    conv1b = tf.Variable(net_pretrained["conv1"]["biases"])
    conv1_in = conv(features, conv1W , conv1b , kernel_size , kernel_size ,channels , stride , stride , padding="SAME" , group = 1)
    conv1 = tf.nn.relu(conv1_in)

    # normalization ?? 
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
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
    conv2W = tf.Variable(net_pretrained["conv2"]["weights"])
    conv2b = tf.Variable(net_pretrained["conv2"]["biases"])
    conv2_in = conv(maxpool1 , conv2W , conv2b , kernel , kernel, channels , stride , stride , padding="SAME", group =group )

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

    #conv3
    kernel = 3
    channels = 384
    stride = 1
    conv3_w = tf.Variable(net_pretrained["conv3"]["weights"])
    conv3_b = tf.Variable(net_pretrained["conv3"]["biases"])
    conv3_in = conv(maxpool2 , conv3_w , conv3_b , kernel , kernel , channels , stride , stride , padding="SAME")
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    kernel = 3
    channels = 384
    stride = 1
    group = 2
    conv4_w = tf.Variable(net_pretrained["conv4"]["weights"])
    conv4_b = tf.Variable(net_pretrained["conv4"]["biases"])
    conv4_in = conv(conv3 , conv4_w , conv4_b , kernel , kernel , channels , stride , stride , padding="SAME" , group = group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    kernel = 3
    channels = 256
    stride = 1
    group = 2
    conv5_w = tf.Variable(net_pretrained["conv5"]["weights"])
    conv5_b = tf.Variable(net_pretrained["conv5"]["biases"])
    conv5_in = conv(conv4 , conv5_w , conv5_b , kernel , kernel , channels, stride , stride , padding="SAME", group =2)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool 5
    kernel = 3
    stride = 3
    maxpool5 = tf.nn.max_pool(conv5 ,ksize = [1 , kernel , kernel , 1] , strides= [1 , stride , stride ,1] , padding= "VALID")


    # fc6
    fc6_w = tf.Variable(net_pretrained["fc6"]["weights"])
    fc6_b = tf.Variable(net_pretrained["fc6"]["biases"])
    full_connected6 = tf.nn.relu_layer(tf.reshape(maxpool5 , [-1 , int(np.prod(maxpool5.get_shape()[1:]))]) , fc6_w, fc6_b)

    #fc7
    fc7_w = tf.Variable(net_pretrained["fc7"]["weights"])
    fc7_b = tf.Variable(net_pretrained["fc7"]["biases"])
    fc7 = tf.nn.relu_layer(full_connected6 , fc7W , fc7b)

    if for_transfer_learning:
        return fc7
    
    fc8_w = tf.Variable(net_pretrained["fc8"]["weights"])
    fc8_b = tf.Variable(net_pretrained["fc8"]["biases"])

    logits = tf.nn.xw_plus_b(fc7 , fc8_w , fc8_b)
    probs = tf.nn.softmax(logits)

    return  probs

