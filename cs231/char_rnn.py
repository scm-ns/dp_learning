
# Copy of Andrej Karpathy min_char_rnn.py cose

import numpy as np

data = open("input.txt" , "r").read()
chars = list(set(data))

# mapping of all the characters to indices.
# used to create one hot encodings
chars_to_index = { char : i for i , char in enumerate(chars) }
index_to_chars = { i : char for i , char in enumerate(chars) }

# hyperparams
hid_layer_size = 100
sequence_len = 25
learning_rate = 1e-1

#model params
weights_xh = np.random.randn(hid_layer_size , len(chars)) * 0.01
weights_hh = np.random.randn(hid_layer_size , hid_layer_size) * 0.01
weights_hy = np.random.randn(len(chars) , hid_layer_size) * 0.01

bias_h = np.zeros((hid_layer_size , 1))
bias_y = np.zeros((len(chars) , 1))




