"""
    Based off of : http://www.cs.toronto.edu/~frossard/post/vgg16/ 
"""


import tensorflow as tf
import numpy as np

class vgg16:
    def __init__(self, imgs):

        self.params = []
        
        # remove imagenet image means
        imagenet_mean = [123.68 , 116.779 , 103.939]
        mean = tf.constant( imagenet_mean , dtype= tf.float32 , shape=[1 , 1 , 1 , 3] , name = "img_mean")
        self.imgs = imgs
        net_in = self.imgs - mean
        self.input_layer = net_in
   
        conv_1_1_out = 0;
        with tf.name_scope("conv_1_1") as scope: 
            in_size = 3;
            out_size = 64;
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(net_in , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_1_1_out = tf.nn.relu(out)
            self.params += [weights , bias]

        self.first_layer = conv_1_1_out;
 
        conv_1_2_out = 0;
        with tf.name_scope("conv_1_2") as scope: 
            in_size = 64;
            out_size = 64
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_1_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_1_2_out = tf.nn.relu(out)
            self.params += [weights , bias]
    

        pool_1 = tf.nn.max_pool(conv_1_2_out , ksize = [1 ,2 , 2 , 1] , strides = [1 ,2 ,2 , 1], padding="SAME" )
        

        conv_2_1_out = 0;
        with tf.name_scope("conv_2_1") as scope: 
            in_size = 64
            out_size = 128
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(pool_1, weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_2_1_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_2_2_out = 0;
        with tf.name_scope("conv_2_2") as scope: 
            in_size = 128
            out_size = 128
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_2_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_2_2_out = tf.nn.relu(out)
            self.params += [weights , bias]

        pool2 = tf.nn.max_pool(conv_2_2_out , ksize = [1 , 2 ,2 , 1] , strides= [1 , 2 ,2 ,1 ], padding = "SAME" )


        conv_3_1_out =0 ;
        with tf.name_scope("conv_3_1") as scope: 
            in_size = 128
            out_size = 256
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(pool_2, weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_3_1_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_3_2_out = 0;
        with tf.name_scope("conv_3_2") as scope: 
            in_size = 256
            out_size = 256
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_3_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_3_2_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_3_3_out = 0;
        with tf.name_scope("conv_3_3") as scope: 
            in_size = 256
            out_size = 256
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_3_2_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_3_3_out = tf.nn.relu(out)
            self.params += [weights , bias]

        pool3 = tf.nn.max_pool(conv_3_3_out , ksize = [1 , 2 ,2 , 1] , strides= [1 , 2 ,2 ,1 ], padding = "SAME" )

        conv_4_1_out = 0;
        with tf.name_scope("conv_4_1") as scope: 
            in_size = 256
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(pool_3, weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_4_1_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_4_2_out = 0;
        with tf.name_scope("conv_4_2") as scope: 
            in_size = 512
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_4_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_4_2_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_4_3_out = 0;
        with tf.name_scope("conv_4_3") as scope: 
            in_size = 512
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_4_2_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_4_3_out = tf.nn.relu(out)
            self.params += [weights , bias]

        pool4 = tf.nn.max_pool(conv_4_3_out , ksize = [1 , 2 ,2 , 1] , strides= [1 , 2 ,2 ,1 ], padding = "SAME" )

        conv_5_1_out = 0;
        with tf.name_scope("conv_5_1") as scope: 
            in_size = 512
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(pool_4, weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_5_1_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_5_2_out = 0 ;
        with tf.name_scope("conv_5_2") as scope: 
            in_size = 512
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_5_1_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_5_2_out = tf.nn.relu(out)
            self.params += [weights , bias]

        conv_5_3_out=0;
        with tf.name_scope("conv_5_3") as scope: 
            in_size = 512
            out_size = 512
            weights = tf.Variable(tf.truncated_normal([3 , 3 , in_size , out_size] , dtype = tf.float32 , stddev = 1e-1))
            bias = tf.Variable(tf.constant(0.0 , shape = [out_size] , dtype = tf.float32))
            conv = tf.nn.conv2d(conv_5_2_out , weights, [1 ,1 , 1 , 1] , padding = "SAME") 
            out = tf.nn.bias_add(conv , bias) 
            conv_5_3_out = tf.nn.relu(out)
            self.params += [weights , bias]

        pool5 = tf.nn.max_pool(conv_5_3_out , ksize = [1 , 2 ,2 , 1] , strides= [1 , 2 ,2 ,1 ], padding = "SAME" )

        fc1 = 0;
        with tf.name_scope("fc1") as scope:
            num_pool_units = int(np.prod(pool5.get_shape()[1:])) # get the number of units in the pool layer # 1: ignore the number of images feed into the network  
            in_size = num_pool_units
            out_size = 4096
            weights = tf.Variable(tf.truncated_normal([in_size , out_size] , dtype= tf.float32))
            bias = tf.Variable(tf.constant(1.0 , shape= [out_size] , dtype = tf.float32))
            pool5_flat = tf.resize(pool5 , [-1 , num_pool_units])
            out = tf.nn.bias_add(tf.nn.matmul(pool5_flat , weights) , bias)
            fc1 = tf.nn.relu(out)
            self.params += [weights , bias]

        fc2 = 0 ;
        with tf.name_scope("fc1") as scope:
            in_size = 4096
            out_size = 4096
            weights = tf.Variable(tf.truncated_normal([in_size , out_size] , dtype= tf.float32))
            bias = tf.Variable(tf.constant(1.0 , shape= [out_size] , dtype = tf.float32))
            out = tf.nn.bias_add(tf.nn.matmul(fc1 , weights) , bias)
            fc2 = tf.nn.relu(out)
            self.params += [weights , bias]

        fc3 = 0;
        with tf.name_scope("fc2") as scope:
            in_size = 4096
            out_size = 4096
            weights = tf.Variable(tf.truncated_normal([in_size , out_size] , dtype= tf.float32))
            bias = tf.Variable(tf.constant(1.0 , shape = [out_size] , dtype = tf.float32))
            out = tf.nn.bias_add(tf.nn.matmul(fc2 , weights) , bias)
            fc3 = out ; # no relu for the last layer # apply softmax to the output
            self.params += [weights , bias]

        self.output_layer = fc3;

    def load_weights(self , weight_file , sess):
            pretrained_weights = np.load(weight_file)
            keys = sorted(pretrained_weights.keys())
            for idx , key in enumerate(keys):
                print idx , key , np.shape(pretrained_weights[key])
                sess.run(self.params[idx].assign(pretrained_weights[key]))

    def predicted_probs_layer():
            return  tf.nn.softmax(self.output_layer);


if __name__ == "__main__":
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32 , [None , 224 , 224 , 3])
    vgg = vgg16(imgs)
    vgg.load_weights("vgg16_weights.npz", sess)
    
    im1 = "file_name"
    im1 = imread(im1 , mode = "RGB" )
    im1 = imresize(im1 ,  ( 224 , 244))
    
    probs = sess.run(vgg.predicted_probs_layer() , feed_dict = {vgg.imgs : [im1]})[0]
    preds = (np.argsort(probs)[::-1][0:5])
    for p in preds: 
        print class_names[p] , prob[p]

