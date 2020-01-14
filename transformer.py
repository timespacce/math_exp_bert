import tensorflow as tf
import numpy as np

from encoder import Encoder


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class Transformer(tf.keras.Model):

    def __init__(self, b_p_gpu, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, rate):
        super(Transformer, self).__init__()

        self.b_p_gpu = b_p_gpu
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.rate = rate

        self.seq_embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)
        self.token_embedding = tf.keras.layers.Embedding(2, self.d_model)
        self.pos_encoding = positional_encoding(260, self.d_model)

        self.pre_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_do = tf.keras.layers.Dropout(rate)

        self.encoder = Encoder(self.num_layers, self.d_model, self.num_heads, self.dff, self.rate)

        self.wh = tf.keras.layers.Dense(self.d_model)
        self.wh_leaky_relu = tf.keras.layers.LeakyReLU(alpha=1e-1)
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

        seq_len = in_seq.shape[1]

        y_hat = self.seq_embedding(in_seq)  # (batch_size, sequence_len, embedding_len)
        y_hat += self.token_embedding(in_seg)  # (batch_size, sequence_len, embedding_len)
        y_hat += self.pos_encoding[:, :seq_len, :]  # (batch_size, sequence_len, embedding_len)

        y_hat = self.pre_bn(y_hat)  # (batch_size, sequence_len, embedding_len)
        y_hat = self.pre_do(y_hat)  # (batch_size, sequence_len, embedding_len)

        y_hat = self.encoder(y_hat, enc_padding_mask)  # (batch_size, sequence_len, embedding_len)

        y_hat_mask = tf.gather(y_hat, mask_indices, batch_dims=1)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.wh(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.wh_leaky_relu(y_hat_mask)
        y_hat_mask = self.post_bn(y_hat_mask)  # (batch_size, mask_len, embedding_len)

        y_hat_mask = tf.matmul(y_hat_mask, self.seq_embedding.weights[0], transpose_b=True)  # (batch_size, mask_len, dictionary_len)
        y_hat_mask = self.sm_seq(y_hat_mask)  # (batch_size, mask_len, dictionary_len)

        y_hat_ns = y_hat[:, 0:1, :]
        y_hat_ns = tf.squeeze(y_hat_ns, axis=1)  # (batch_size, 1, hidden_size)
        y_hat_ns = self.wns(y_hat_ns)  # (batch_size, 1, 2)

        # @TODO CHECK!!

        y_hat_ns = self.sm_ns(y_hat_ns)  # (batch_size, 1, 2)

        return y_hat_mask, y_hat_ns
