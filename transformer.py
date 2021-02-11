<<<<<<< HEAD
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
        # EMBEDDING
        self.bert_embedding = BERT_Embedding(self.input_vocab_size, self.hidden_size, self.target_vocab_size, self.max_seq_len)
        # NORMALIZATION
        self.pre_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_do = tf.keras.layers.Dropout(rate)
        # ENCODING
        self.encoder = Encoder(self.num_layers, self.hidden_size, self.intermediate_size, self.num_heads, self.rate)
        # MASKING
        self.wh = tf.keras.layers.Dense(self.hidden_size)
        # self.activation = tf.keras.layers.LeakyReLU(alpha=1e-1)
        self.activation = GeLU()
        self.post_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.sm_seq = tf.keras.layers.Softmax(axis=-1)
        # CLASSIFICATION
        # self.wsp = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.wns = tf.keras.layers.Dense(2)
        self.sm_ns = tf.keras.layers.Softmax(axis=-1)
        return

    def classify(self, x_enc, mask_indices=None, mode=None):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mask_indices: (batch_size, sequence_len)
            mode: 'GENERATIVE' or 'DISCRIMINATIVE'

        Returns:

        """

        # CLASSIFICATION
        # return self.pre_train_classify(x_enc=x_enc, mask_indices=mask_indices)
        return self.fine_tune_classify(x_enc=x_enc, mode=mode)

    def pre_train_classify(self, x_enc, mask_indices):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mask_indices: (batch_size, sequence_len)

        Returns:

        """
        # MASKING
        y_hat_mask = tf.gather(x_enc, mask_indices, batch_dims=1)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.wh(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.activation(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.post_bn(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        word_embedding_weights = self.bert_embedding.get_word_embeddings()
        y_hat_mask = tf.matmul(y_hat_mask, word_embedding_weights, transpose_b=True)  # (batch_size, mask_len, dictionary_len)
        y_hat_mask = self.sm_seq(y_hat_mask)  # (batch_size, mask_len, dictionary_len)
        # CLASSIFICATION
        y_hat_sp = x_enc[:, 0:1, :]  # (batch_size, 1, hidden_size)
        y_hat_sp = tf.squeeze(y_hat_sp, axis=1)  # (batch_size, hidden_size)
        # y_hat_sp = self.wsp(y_hat_sp)  # (batch_size, hidden_size)
        y_hat_sp = self.wns(y_hat_sp)  # (batch_size, 1, 2)
        y_hat_sp = self.sm_ns(y_hat_sp)  # (batch_size, 1, 2)
        ##
        return y_hat_mask, y_hat_sp

    def fine_tune_classify(self, x_enc, mode):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mode: 'GENERATIVE' or 'DISCRIMINATIVE'

        Returns:

        """
        y_hat = x_enc

        if mode == 'GENERATIVE':
            word_embedding_weights = self.bert_embedding.get_word_embeddings()
            y_hat = tf.matmul(y_hat, word_embedding_weights, transpose_b=True)  # (batch_size, sequence_len, dictionary_len)
            y_hat = self.sm_seq(y_hat)  # (batch_size, sequence_len, dictionary_len)

        if mode == 'DISCRIMINATIVE':
            y_hat = y_hat[:, 0:1, :]  # (batch_size, 1, hidden_size)
            y_hat = self.wns(y_hat)  # (batch_size, 1, 2)
            y_hat = self.sm_ns(y_hat)  # (batch_size, 1, 2)

        return y_hat

    def call(self, in_seq, enc_padding_mask, in_seg):
        """

        Args:
            in_seq:             (batch_size, sequence_len)
            enc_padding_mask:   (batch_size, sequence_len)
            in_seg:             (batch_size, sequence_len)

        Returns:

        """

        # EMBEDDING
        x_emb = self.bert_embedding(in_seq, in_seg)  # (batch_size, sequence_len, hidden_size)
        # ENCODING
        x_emb = self.pre_bn(x_emb)  # (batch_size, sequence_len, hidden_size)
        x_emb = self.pre_do(x_emb)  # (batch_size, sequence_len, hidden_size)
        x_enc = self.encoder(x_emb, enc_padding_mask)  # (batch_size, sequence_len, hidden_size)
        ##
        return x_enc
=======
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
        # EMBEDDING
        self.bert_embedding = BERT_Embedding(self.input_vocab_size, self.hidden_size, self.target_vocab_size, self.max_seq_len)
        # NORMALIZATION
        self.pre_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.pre_do = tf.keras.layers.Dropout(rate)
        # ENCODING
        self.encoder = Encoder(self.num_layers, self.hidden_size, self.intermediate_size, self.num_heads, self.rate)
        # MASKING
        self.wh = tf.keras.layers.Dense(self.hidden_size)
        # self.activation = tf.keras.layers.LeakyReLU(alpha=1e-1)
        self.activation = GeLU()
        self.post_bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.sm_seq = tf.keras.layers.Softmax(axis=-1)
        # CLASSIFICATION
        # self.wsp = tf.keras.layers.Dense(hidden_size, activation='tanh')
        self.wns = tf.keras.layers.Dense(2)
        self.sm_ns = tf.keras.layers.Softmax(axis=-1)
        return

    def classify(self, x_enc, mask_indices=None, mode=None):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mask_indices: (batch_size, sequence_len)
            mode: 'GENERATIVE' or 'DISCRIMINATIVE'

        Returns:

        """

        # CLASSIFICATION
        # return self.pre_train_classify(x_enc=x_enc, mask_indices=mask_indices)
        return self.fine_tune_classify(x_enc=x_enc, mode=mode)

    def pre_train_classify(self, x_enc, mask_indices):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mask_indices: (batch_size, sequence_len)

        Returns:

        """
        # MASKING
        y_hat_mask = tf.gather(x_enc, mask_indices, batch_dims=1)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.wh(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.activation(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        y_hat_mask = self.post_bn(y_hat_mask)  # (batch_size, mask_len, embedding_len)
        word_embedding_weights = self.bert_embedding.get_word_embeddings()
        y_hat_mask = tf.matmul(y_hat_mask, word_embedding_weights, transpose_b=True)  # (batch_size, mask_len, dictionary_len)
        y_hat_mask = self.sm_seq(y_hat_mask)  # (batch_size, mask_len, dictionary_len)
        # CLASSIFICATION
        y_hat_sp = x_enc[:, 0:1, :]  # (batch_size, 1, hidden_size)
        y_hat_sp = tf.squeeze(y_hat_sp, axis=1)  # (batch_size, hidden_size)
        # y_hat_sp = self.wsp(y_hat_sp)  # (batch_size, hidden_size)
        y_hat_sp = self.wns(y_hat_sp)  # (batch_size, 1, 2)
        y_hat_sp = self.sm_ns(y_hat_sp)  # (batch_size, 1, 2)
        ##
        return y_hat_mask, y_hat_sp

    def fine_tune_classify(self, x_enc, mode):
        """

        Args:
            x_enc: (batch_size, sequence_len, embedding_len)
            mode: 'GENERATIVE' or 'DISCRIMINATIVE'

        Returns:

        """
        y_hat = x_enc

        if mode == 'GENERATIVE':
            word_embedding_weights = self.bert_embedding.get_word_embeddings()
            y_hat = tf.matmul(y_hat, word_embedding_weights, transpose_b=True)  # (batch_size, sequence_len, dictionary_len)
            y_hat = self.sm_seq(y_hat)  # (batch_size, sequence_len, dictionary_len)

        if mode == 'DISCRIMINATIVE':
            y_hat = y_hat[:, 0:1, :]  # (batch_size, 1, hidden_size)
            y_hat = self.wns(y_hat)  # (batch_size, 1, 2)
            y_hat = self.sm_ns(y_hat)  # (batch_size, 1, 2)

        return y_hat

    def call(self, in_seq, enc_padding_mask, in_seg):
        """

        Args:
            in_seq:             (batch_size, sequence_len)
            enc_padding_mask:   (batch_size, sequence_len)
            in_seg:             (batch_size, sequence_len)

        Returns:

        """

        # EMBEDDING
        x_emb = self.bert_embedding(in_seq, in_seg)  # (batch_size, sequence_len, hidden_size)
        # ENCODING
        x_emb = self.pre_bn(x_emb)  # (batch_size, sequence_len, hidden_size)
        x_emb = self.pre_do(x_emb)  # (batch_size, sequence_len, hidden_size)
        x_enc = self.encoder(x_emb, enc_padding_mask)  # (batch_size, sequence_len, hidden_size)
        ##
        return x_enc
>>>>>>> 4529c5efa64afdb1171f4faddc4aa8535fd31559
