import tensorflow as tf

from bert_embedding import BERT_Embedding
from encoder import Encoder
from gelu import GeLU


class Transformer(tf.keras.Model):

    def __init__(self, max_seq_len, b_p_gpu, num_layers, hidden_size, intermediate_size, num_heads, input_vocab_size, target_vocab_size,
                 rate):
        super(Transformer, self).__init__()

        self.max_seq_len = max_seq_len
        self.b_p_gpu = b_p_gpu
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.rate = rate

        self.bert_embedding = BERT_Embedding(self.input_vocab_size, self.hidden_size, self.target_vocab_size, self.max_seq_len)

        self.pre_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_do = tf.keras.layers.Dropout(rate)

        self.encoder = Encoder(self.num_layers, self.hidden_size, self.intermediate_size, self.num_heads, self.rate)

        self.wh = tf.keras.layers.Dense(self.hidden_size)
        # self.activation = tf.keras.layers.LeakyReLU(alpha=1e-1)
        self.activation = GeLU()
        self.post_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.sm_seq = tf.keras.layers.Softmax(axis=-1)

        self.wns = tf.keras.layers.Dense(2, activation='tanh')
        self.sm_ns = tf.keras.layers.Softmax(axis=-1)
        return

    def call(self, in_seq, enc_padding_mask, in_seg, mask_indices):
        """

        Args:
            in_seq:             (batch_size, sequence_len)
            enc_padding_mask:   (batch_size, sequence_len)
            in_seg:             (batch_size, sequence_len)
            mask_indices:       (batch_size, sequence_len)

        Returns:

        """

        y_hat = self.bert_embedding(in_seq, in_seg)

        y_hat = self.pre_bn(y_hat)  # (batch_size, sequence_len, embedding_len)
        y_hat = self.pre_do(y_hat)  # (batch_size, sequence_len, embedding_len)

        y_hat = self.encoder(y_hat, enc_padding_mask)  # (batch_size, sequence_len, embedding_len)

        y_hat_mask = tf.gather(y_hat, mask_indices, batch_dims=1)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.wh(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.activation(y_hat_mask)
        y_hat_mask = self.post_bn(y_hat_mask)  # (batch_size, mask_len, embedding_len)

        y_hat_mask = tf.matmul(y_hat_mask, self.bert_embedding.get_word_embeddings(), transpose_b=True)
        # (batch_size, mask_len, dictionary_len)
        y_hat_mask = self.sm_seq(y_hat_mask)  # (batch_size, mask_len, dictionary_len)

        y_hat_ns = y_hat[:, 0:1, :]
        y_hat_ns = tf.squeeze(y_hat_ns, axis=1)  # (batch_size, 1, hidden_size)
        y_hat_ns = self.wns(y_hat_ns)  # (batch_size, 1, 2)
        y_hat_ns = self.sm_ns(y_hat_ns)  # (batch_size, 1, 2)

        return y_hat_mask, y_hat_ns
