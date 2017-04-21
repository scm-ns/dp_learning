# Copy/Rewrite of fast.ai tutorials

from theano.sandbox import cuda

from utils import *
from __future__ import division, print_function


path = "data/dogscats/"
modelpath = path + "models/"

if not os.path.exists(model_path):
    os.mkdir(model_path)

batch_size = 64



model = vgg_ft(2)


model.load_weights(model_path + "finetine3.h5")


layers = model.layers

"""
    Dropout -> underfitting on the training set as we randomly setting the weights to zero
    Try removing dropout to get better training accuracy. 
    Droput applied to full connected network
    So freeze rest of the network with the weights that were pretrained.
            ie . convolution layer weights are not updated.
    Then retrain on the fully connected networks without the dropout  
"""


# Get all the convolution layers and then get the last layer
last_conv_idx = [index for index,layer in enumerate(layers) if type(layer) is Convolutional2D][-1]

print(last_conv_idx)


layers[last_conv_idx]

conv_layers = layers[:last_conv_idx + 1]
conv_model = Sequential(conv_layers) # Create seperate model from only the conv 2d layers

fc_layers = layers[last_conv_idx + 1:] # Create fc starting from the end of the conv2d layers




