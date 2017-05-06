
import os
import collections
from six.moves import cPickle
import numpy as np

class text_loader():
    def __init__(self , data_dir , batch_size , seq_len , encoding = "utf-8")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.encoding = encoding

        input_file = os.path.join(data_dir , "input.txt")
        vocab_file = os.path.join(data_dir , "vocab.pkl")
        tensor_file = os.path.join(data_dir , "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            self.process_text(input_file , vocab_file , tensor_file)
        else:
            self.load_preprocessed(vocab_file , tensor_file)

        self.create_batches()
        self.reset_batch_pointer()


    def preprocess(self , input_file , vc_f , tf):
        with codecs.open(input_file , "r" , encoding=self.encoding) as f:
            data = f.read()

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items() , key=lambda x : -x[1]) # extracts the count from most common to least common

        self.chars , _  = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars , range(len(self.chars)))) # dictionary of (chars , idx) in the decresing order, but the idx starts from 0 on. so 0 idx -> most common char
        with open(vocab_file , "wb") as f:
            cPickle.dump(self.chars , f)

        # till now we are encoding the charsters as indexes starting from the most common to the least common

        # so now we have an encoding of each character, so instead of storing the characters, we just have to
        # store the index of that character
        # the map function, maps the character from the data into the get function for the dict storing the
        # character - index encoding , and gets the index as the output
        self.tensor = np.array(list(map(self.vocab.get , data))) # store the same char data in a tensor
        # self.tensor stores a list of characters, not stored as chars but as ints. (as an numpy array)
        # So now we are in a good format to feed it into the reccurent network
        
        np.save(tensor_file , self.tensor)


    def load_preprocessed(self, vocab_file , tensor_file):
        with open(vocab_file , "rb") as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars , range(len(chars))))
        self.tensor = np.load(tensor_file)
        # 1 batch is batch_size * seq_len number of characters feed into the network
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_len))


