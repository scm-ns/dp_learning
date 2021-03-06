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
        input_groups = tf.split(input , group ,3) # how to divide the input into groups with
        kernel_groups = tf.split(kernel , group , 3)
        output_groups = [convolve(i,k) for i , k in zip(input_groups , kernel_groups)]
        conv = tf.concat( output_groups , 3)
    # reshape the tensor, may be for feeding or reading from caffe ??
    return tf.reshape(tf.nn.bias_add(conv , biases) , [-1] + conv.get_shape().as_list()[1:])


def alex_net(features , retrain_last_layer_num_classes , for_transfer_learning = False ):
    # Builds model and loads pretrained model
    kernel_size = 11
    stride = 4
    channels = 96
    conv1W = tf.Variable(net_pretrained["conv1"][0])
    conv1b = tf.Variable(net_pretrained["conv1"][1])
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
    maxpool1 = tf.nn.max_pool(lrn1 , ksize=[1 , kernel , kernel , 1] , strides=[1 , stride , stride , 1] , padding =padding)

    # conv2
    kernel = 5
    channels = 256
    stride = 1
    group = 2
    conv2W = tf.Variable(net_pretrained["conv2"][0])
    conv2b = tf.Variable(net_pretrained["conv2"][1])
    conv2_in = conv(maxpool1 , conv2W , conv2b , kernel , kernel, channels , stride , stride , padding="SAME", group =group )
    conv2 = tf.nn.relu(conv2_in)

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
    conv3_w = tf.Variable(net_pretrained["conv3"][0])
    conv3_b = tf.Variable(net_pretrained["conv3"][1])
    conv3_in = conv(maxpool2 , conv3_w , conv3_b , kernel , kernel , channels , stride , stride , padding="SAME")
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    kernel = 3
    channels = 384
    stride = 1
    group = 2
    conv4_w = tf.Variable(net_pretrained["conv4"][0])
    conv4_b = tf.Variable(net_pretrained["conv4"][1])
    conv4_in = conv(conv3 , conv4_w , conv4_b , kernel , kernel , channels , stride , stride , padding="SAME" , group = group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    kernel = 3
    channels = 256
    stride = 1
    conv5_w = tf.Variable(net_pretrained["conv5"][0])
    conv5_b = tf.Variable(net_pretrained["conv5"][1])
    conv5_in = conv(conv4 , conv5_w , conv5_b , kernel , kernel , channels, stride , stride , padding="SAME", group =2)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool 5
    kernel = 3
    stride = 2
    maxpool5 = tf.nn.max_pool(conv5 ,ksize = [1 , kernel , kernel , 1] , strides= [1 , stride , stride ,1] , padding= "VALID")


    # fc6
    fc6_w = tf.Variable(net_pretrained["fc6"][0])
    fc6_b = tf.Variable(net_pretrained["fc6"][1])
    full_connected6 = tf.nn.relu_layer(tf.reshape(maxpool5 , [-1 , int(np.prod(maxpool5.get_shape()[1:]))]) , fc6_w, fc6_b)

    #fc7
    fc7_w = tf.Variable(net_pretrained["fc7"][0])
    fc7_b = tf.Variable(net_pretrained["fc7"][1])
    fc7 = tf.nn.relu_layer(full_connected6 , fc7_w , fc7_b)

    if for_transfer_learning:
        return fc7

    if retrain_last_layer_num_classes is None:
        fc8_w = tf.Variable(net_pretrained["fc8"][0])
        fc8_b = tf.Variable(net_pretrained["fc8"][1])
        logits = tf.nn.xw_plus_b(fc7 , fc8_w , fc8_b)
    else :
        fc8_w = tf.Variable(tf.truncated_normal([net_pretrained["fc8"][0].shape[0] , retrain_last_layer_num_classes] , stddev = 0.005))
        fc8_b = tf.Variable(tf.zeros(retrain_last_layer_num_classes))
        logits = tf.nn.xw_plus_b(fc7 , fc8_w , fc8_b)
        
    return tf.nn.softmax(logits)
