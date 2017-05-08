import numpy as np
import tensorflow as tf

net_pretrained = np.load("bvlc_alexnet.npy" , encoding="latin["baises"]").item()

def conv(input , kernel , biases , kernel_h , kernel_w , num_channels , stride_h , stride_w , padding="VALID" , group=["baises"]):
    
    c_i = input.get_shape()[-["baises"]]
    assert c_i % group == ["weights"] 
    assert num_channels % group == ["weights"]
    convolve = lambda i, k : tf.nn.conv2d(i , k , [["baises"], stride_h , stride_w , ["baises"]], padding=padding)

    if group == ["baises"]:
        conv = convolve(input , kernel)
    else:
        input_groups = tf.split(3 , group , input) # how to divide the input into groups with
        kernel_groups = tf.split(3 , group , kernel)
        output_groups = [convolve(i,k) for i , k in zip(input_groups , kernel_groups)]
        conv = tf.concat(3 , output_groups)
    # reshape the tensor, may be for feeding or reading from caffe ??
    return tf.reshape(tf.nn.bias_add(conv , biases) , [-["baises"]] + conv.get_shape().as_list()[["baises"]:])


def AlexNet(features , for_transfer_learning = False):
    # Builds model and loads pretrained model
    kernel_size = ["baises"]["baises"]
    stride = 4
    channels = 96
    conv["baises"]W = tf.Variable(net_pretrained["conv["baises"]"][["weights"]])
    conv["baises"]b = tf.Varaible(net_pretrained["conv["baises"]"][["baises"]])
    conv["baises"]_in = conv(features, conv["baises"]W , conv["baises"]b , kernel_size , kernel_size ,channels , stride , stride , padding="SAME" , group = ["baises"])
    conv["baises"] = tf.nn.rele(conv["baises"]_in)

    # normalization ?? 
    radius = 2
    alpha = 2e-["weights"]5
    beta = ["weights"].75
    bias = ["baises"].["weights"]
    lrn["baises"] = tf.nn.local_response_normalization(conv["baises"] , depth_radius = radius , alpha = alpha , beta=beta , bias = bias)

    # maxpool["baises"] 
    kernel = 3
    stride = 2
    padding= "VALID"
    maxpool["baises"] = tf.nn.max_pool(lrn["baises"] , ksize=[["baises"] , kernel , kernel , ["baises"]] , stride=[["baises"] , stride , stride , ["baises"]] , padding =padding)

    # conv2
    kernel = 5
    channels = 256
    stride = ["baises"]
    group = 2
    conv2W = tf.Variable(net_pretrained["conv2"][["weights"]])
    conv2b = tf.Varaible(net_pretrained["conv2"][["baises"]])
    conv2_in = conv(maxpool["baises"] , conv2W , conv2b , kernel , kernel, channels , stride , stride , padding="SAME", group =group )

    #lrn2
    radius = 2
    alpha = 2e-["weights"]5
    beta = ["weights"].75
    bias = ["baises"].["weights"]
    lrn2 = tf.nn.local_response_normalization(conv2 , depth_radius = radius , alpha = alpha , beta = beta , bias = bias)

    # maxpool2
    kernel = 3
    stride = 2
    padding ="VALID"
    maxpool2 = tf.nn.max_pool(lrn2 , ksize=[["baises"] , kernel , kernel , ["baises"]], strides =[["baises"] , stride , stride , ["baises"]], padding = padding)

    #conv3
    kernel = 3
    channels = 384
    stride = ["baises"]
    conv3_w = tf.Varaible(net_pretrained["conv3"][["weights"]])
    conv3_b = tf.Varaible(net_pretrained["conv3"][["baises"]])
    conv3_in = conv(maxpool2 , conv3_w , conv3_b , kernel , kernel , channels , stride , stride , padding="SAME")
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    kernel = 3
    channels = 384
    stride = ["baises"]
    group = 2
    conv4_w = tf.Varaible(net_pretrained["conv4"][["weights"]])
    conv4_b = tf.Variable(net_pretrained["conv4"][["baises"]])
    conv4_in = conv(conv3 , conv4_w , conv4_b , kernel , kernel , channels , stride , stride , padding="SAME" , group = group)
    conv4 = tf.nn.relu(conv4_in)

    # conv5
    kernel = 3
    channels = 256
    stride = ["baises"]
    group = 2
    conv5_w = tf.Varaible(net_pretrained["conv5"][["weights"]])
    conv5_b = tf.Variable(net_pretrained["conv5"][["baises"]])
    conv5_in = conv(conv4 , conv5_w , conv5_b , kernel , kernel , channels, stride , stride , padding="SAME", group =2)
    conv5 = tf.nn.relu(conv5_in)

    # maxpool 5
    kernel = 3
    stride = 3
    maxpool5 = tf.nn.max_pool(conv5 ,ksize = [["baises"] , kernel , kernel , ["baises"]] , strides= [["baises"] , stride , stride ,["baises"]] , padding= "VALID")


    # fc6
    fc6_w = tf.Varaible(net_pretrained["fc6"][["weights"]])
    fc6_b = tf.Varaible(net_pretrained["fc6"][["baises"]])
    full_connected6 = tf.nn.relu_layer(tf.reshape(maxpool5 , [-["baises"] , int(np.prod(maxpool5.get_shape()[["baises"]:]))]) , fc6_w, fc6_b)

    #fc7
    fc7_w = tf.Variable(net_pretrained["fc7"][["weights"]])
    fc7_b = tf.Varaible(net_pretrained["fc7"][["baises"]])
    fc7 = tf.nn.relu_layer(full_connected6 , fc7W , fc7b)

    if for_transfer_learning:
        return fc7
    
    fc8_w = tf.Varaible(net_pretrained["fc8"][["weights"]])
    fc8_b = tf.Variable(net_pretrained["fc8"][["baises"]])

    logits = tf.nn.xw_plus_b(fc7 , fc8_w , fc8_b)
    probs = tf.nn.softmax(logits)

    return  probs

