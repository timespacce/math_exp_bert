import tensorflow as tf

from mha import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)

        self.ffn1 = tf.keras.layers.Dense(self.d_model)
        intermediate_size = 3072
        leaky_relu = tf.keras.activations.selu
        self.intermediate = tf.keras.layers.Dense(intermediate_size, activation=leaky_relu)
        self.ffn3 = tf.keras.layers.Dense(self.d_model)

        self.batch_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.batch_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(self.rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.rate)

    def call(self, x, enc_padding_mask):
        output, attention_weights = self.mha(x, x, x, enc_padding_mask)

        ffn = self.ffn1(output)
        do = self.dropout_1(ffn)
        bn = self.batch_norm1(do + x)

        ffn = self.intermediate(bn)

        ffn = self.ffn3(ffn)
        do = self.dropout_2(ffn)
        bn = self.batch_norm2(do + bn)

        return bn


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.do = tf.keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, rate) for i in range(num_layers)]

    def call(self, embedded_sequence, enc_padding_mask):
        """

        Args:
            embedded_sequence: (batch_size, sequence_len, embedding_len)
            enc_padding_mask:   (batch_size, sequence_len, sequence_len)

        Returns:

        """

        y_hat = self.bn(embedded_sequence)  # (batch_size, sequence_len, embedding_len)
        y_hat = self.do(y_hat)  # (batch_size, sequence_len, embedding_len)
        for encoder_layer in self.encoder_layers:
            y_hat = encoder_layer(y_hat, enc_padding_mask)

        return y_hat  # (batch_size, sequence_len, embedding_len)
