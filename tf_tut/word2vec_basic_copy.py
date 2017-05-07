# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename) # COOL!
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
  count = [['UNK', -1]]
  # store the most common words in count. 
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  # creating mapping between a word and index, using the current lenght of dict as the index
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary: # If word amount the common
      index = dictionary[word] 
    else:
      index = 0  # dictionary['UNK'] #Howis the zero handled ? 
      unk_count += 1
    # convert the words into indices so that they can be feed into the algorithm
    # but how is the 0 being to map all non common words ? 
    data.append(index) 
  count[0][1] = unk_count # get the number of non mapped( not amont most common) words. Why ??

  # inverted key -value pairs. So instread of word - index mapping, index - word mapping 
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

  return data, count, dictionary, reverse_dictionary

# data : list of indices mapped from words. So sequence of indices instread of sequence of words
# count : list of list, where the each inner list is each word - # occurances
# dict : dict of word - idx mapping
# rev*_dict : dict of idx - word mapping
data, count, dictionary, reverse_dictionary = build_dataset(words) # GET THE DATA !!!!!!!!!!!!!!
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5]) 
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])  # data has no ordering
# so just the first 10 words and the idx for those words

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window): # skip window used to create the context
  global data_index
  assert batch_size % num_skips == 0 # What is num skips ? 
  # context_left ... target ... context_right
  # |       -           |
  #       skip_window 
  # |                   -                   |
  #                 num_skips ?? 
  assert num_skips <= 2 * skip_window # number to skip to the next target context pair ? 
  ## How exactly does an ndarray store data ? 
  # In a tree based structure it seems. 
  """
    say np.ndarray((2 , 3 , 4))

                                *
        
                        /               \
                     *                    *                                     -  2

                /    |     \         /    |     \  
               *     *      *       *     *      *                              - 3 
                                    
               1     1      1       1     1      1                           |
               2     2      2       2     2      2                           |
               3     3      3       3     3      3                           |   
               4     4      4       4     4      4                           |  - 4

    The things like sqeeze operate on this tree strucutre

  """
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  span = 2 * skip_window + 1  # [ skip_window target skip_window ] 

  buffer = collections.deque(maxlen=span) # words to operate ?

  for _ in range(span):
    buffer.append(data[data_index]) # data holds seqence of indices , get an index
    data_index = (data_index + 1) % len(data) # rotating  pointer into data

  for i in range(batch_size // num_skips): # number of context - traget pairs. 
      # so after operating on
      # first skip len number of words move to the next skip len number of words
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window] #index of the target

    for j in range(num_skips): # now process each of the words in the single skip len

      while target in targets_to_avoid: # this is stupid code. Overwrite value in loop ? 
        target = random.randint(0, span - 1) # random variable ??
      targets_to_avoid.append(target) # get the random idx ? 
       # FOr the skip gram model try to predict the context given the target. 
       #
       # the skip_window remains the same for j iteration
       # but the context can point to any other of the word in the buffer
       #    it can even point to the target as the index is random. There is no point in the target_to_avoid thing.
      batch[i * num_skips + j] = buffer[skip_window]  # get the traget
      labels[i * num_skips + j, 0] = buffer[target] # get the onde of the context

    buffer.append(data[data_index]) # why append to the buffer here ?
    # Okay may be for the iter over the next num_skip words
    # so the first loop might have been for handling the initial data. Why not move this to the top 
    # and get rid of this code copy ? 
    # Nope wrong me. This just adds a single word to the buffer
    # may be indxing into the buffer gives the latest items added
    data_index = (data_index + 1) % len(data)
  return batch, labels # create a single batch ?? yep 1 batch = 1 batch size # elems

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]], # print out the words correponding to the index
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label. ?? 
# Okay wrong assmption about num_skips. The division of the batch_size by the num_skips is  
# just to make sure that when we create num_skips labels from a single target, we don't want to
# add more that the batch_size number of items to the batch

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size]) # batch_size * N input
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'): # Cool easy way to swtich between cpu and gpu !!!
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) # Single embedding for each word

    # mapping between the index in the train_input and the embedding vector
    # the idx are distributed along the different embeddings. 
    # the embedidngs are looped up using the idx and so each word will have an embedding
    tf.nn.embedding_lookup
    embed = tf.nn.embedding_lookup(embeddings, train_inputs) # what does this do ? 
    # list of embedding correspoinding to each word

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled, # discriminate between the target embedding and 64 other negative sample embeding. 
                     num_classes=vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  # The algorithm will change the embeddings., we normalize it
  # 
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup( # get the embedding for each of the word in the dataset
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True) # compute similarity between all the embeddings and the validation embedding

  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval() # get the similarity between the validation set word's embeddings and all embeddings
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1] # get embedding, but not of the valid word 
        # each i corresponds to each row with valid_examples number of elements, with each value
        # indicating the similiart between that valid example and embedding (before the multiplication) of that row.

        # sim more similar will have larger values, so sort with negative to get list of most similar to least
        log_str = "Nearest to %s:" % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]] #close words using the similar embeddings.
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)

  final_embeddings = normalized_embeddings.eval()

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :]) # plot only 500 words.
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
