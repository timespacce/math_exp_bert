import tensorflow as tf
import numpy as np


def get_angles(pos, i, hidden_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(hidden_size))
    return pos * angle_rates


def positional_encoding(max_seq_len, hidden_size):
    angle_rads = get_angles(np.arange(max_seq_len)[:, np.newaxis], np.arange(hidden_size)[np.newaxis, :], hidden_size)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class BERT_Embedding(tf.keras.layers.Layer):
    vocab_size = None

    hidden_size = None

    output_size = None

    max_seq_len = None

    mean = None

    std_dev = None

    word_embeddings = None

    positional_encoding = None

    type_embeddings = None

    def __init__(self, vocab_size, hidden_size, output_size, max_seq_len, **kwargs):
        super(BERT_Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_seq_len = max_seq_len

        self.mean = 0.0
        self.std_dev = 0.02
        self.initializer = tf.keras.initializers.TruncatedNormal(mean=self.mean, stddev=self.std_dev)

    def build(self, input_shape):
        self.word_embeddings = self.add_weight(name="w_e", shape=[self.vocab_size, self.hidden_size], initializer=self.initializer)
        self.positional_encoding = self.add_weight(name="p_e", shape=[self.max_seq_len, self.hidden_size], initializer=self.initializer)
        self.type_embeddings = self.add_weight(name="t_e", shape=[self.output_size, self.hidden_size], initializer=self.initializer)

        # self.word_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        # self.type_embeddings = tf.keras.layers.Embedding(self.output_size, self.hidden_size)
        # self.positional_encoding = positional_encoding(self.max_seq_len, self.hidden_size)
        super(BERT_Embedding, self).build(input_shape)

    def get_config(self):
        return {'vocab_size' : self.vocab_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'max_seq_len': self.max_seq_len}

    def get_word_embeddings(self):
        w = self.word_embeddings
        # w = self.word_embeddings.weights[0]
        return w

    def preprocess_custom(self, in_seq, in_seg):
        seq_len = in_seq.shape[1]

        y_hat = self.word_embeddings(in_seq)  # (batch_size, sequence_len, hidden_size)
        y_hat += self.type_embeddings(in_seg)  # (batch_size, sequence_len, hidden_size)
        y_hat += self.positional_encoding[:, :seq_len, :]  # (batch_size, sequence_len, hidden_size)

        return y_hat

    def preprocess_google(self, in_seq, in_seg):
        seq_len = in_seq.shape[1]

        y_hat = tf.gather(self.word_embeddings, tf.cast(in_seq, tf.int32))
        y_hat += self.positional_encoding[:seq_len, :]
        in_seg_one_hot = tf.one_hot(in_seg, 2)
        y_hat += tf.matmul(in_seg_one_hot, self.type_embeddings)
        return y_hat

    def call(self, in_seq, in_seg):
        embedded = self.preprocess_google(in_seq, in_seg)

        return embedded
