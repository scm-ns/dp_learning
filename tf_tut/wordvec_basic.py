# Rewrite + understanding of tensorflow word2vec basic code


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf

url = "http://mattmahoney.net/dc"


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrive(url + filename , filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified" , filename)
    else:
        print(statinfo.st_size)
        raise Execption("Failed to verify")

    return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words  = read_data(filename)
print("Data size" , len(words))

vocab_size = 50000

def build_dataset(words):
    count = [["UNK" , -1]]
    count.extend(collections.Counter(words).most_common(vocab_size -1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) # this does not make sense. May be the dict grows according to insertion
        # and that sets the index
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:


