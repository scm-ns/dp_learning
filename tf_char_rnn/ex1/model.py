import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class model():
    def __init__(self , args , infer = False):
        self.args = args

        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == "rnn":
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == "gru" :
            cell_fn = rnn_cell.GRUCell
        elif args.model == "lstm":
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not support: {}".format(args.model))

        cell = cell_fn(args.rnn_size , state_is_tuple=True) # so multiple rnns ? 
        self.cell = cell =  rnn_cell.MultiRNNCell([cell] * args.num_layers , state_is_tuple=True)

        self.input_data = tf.placeholder(tf.int32 , [args.batch_size , args.seq_length])
       # traget must be a shited ahead input
        self.targets = tf.placeholder(tf.int32 , [args.batch_size , args.seq_length])

        self.initial_state = cell.zero_state(args.batch_size , tf.float32)

        with tf.variable_scope("rnn_lm"):
            softmax_w = tf.get_variable("softmax_w" , [args.rnn_size , args.vocab_size])
            softmax_b = tf.get_variable("softmax_b" , [args.vocab_size])

            with tf.device("/cpu:0"):
                embeddings = tf.get_variable("embeddings" , [args.vocab_size , args.rnn_size])

                inputs = tf.split(1 , args.seq_length , tf.nn.embedding_lookup(emdedding, self.input_data))
                inputs = [tf.squeeze(input_ , [1]) for input_ in inputs] 

        def loop(prev , _):
            prev = tf.matmul(prev , softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev , 1))
            return tf.nn.embedding_lookup(embedding , prev_symbol)

        outputs , last_state = seq2seq,rnn_decode(inputs , self.initial_state , cell , loop_function =loop if infer else None , scope = "rnnlm")

        output = tf.reshape(tf.concat(1 , outputs) , [-1 , args.rnn_size])
        self.logits = tf.matmul(output , softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits] , 
                [tf.reshape(self.targets , [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0 , trainable = False)
        tvars = tf.trainable_variables()
        grads , _ = tf.clip_by_global_norm(tf.gradinets





