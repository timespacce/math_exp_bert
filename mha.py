import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask=None):
    """

    Args:
        q:                  (batch_size, seq_len_q, emb_w)
        k:                  (batch_size, seq_len_k, emb_w)
        v:                  (batch_size, seq_len_k, emb_w)
        mask:               (batch_size, seq_len_q, seq_len_k)

    Returns:
        attention_weights:  (batch-size, seq_len_q, seq_len_k)
        output:             (batch-size, seq_len_q, emb_w)
    """

    # = (batch_size, seq_len_q, emb_w) * transpose((batch_size, seq_len_k, emb_w), axes=[0, 2, 1])
    # = (batch_size, seq_len_q, emb_w * (batch_size, emb_w, seq_len_k)
    # = (batch_size, seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        expanded_mask = tf.expand_dims(mask, axis=[1])
        scaled_attention_logits += (expanded_mask * -1e9)

    # = (batch-size, seq_len_q, seq_len_k) . sum(:, axes=[2]) = (batch_size, seq_len_q, 1)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # = (batch-size, seq_len_q, seq_len_k) * (batch_size, seq_len_k, emb_w)
    # = (batch-size, seq_len_q, emb_w)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # W_q.shape = (seq_len, d_model)
        self.wk = tf.keras.layers.Dense(d_model)  # W_k.shape = (seq_len, d_model)
        self.wv = tf.keras.layers.Dense(d_model)  # W_v.shape = (seq_len, d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """

        Args:
            x:          (batch_size, seq_w, d_model)
            batch_size:

        Returns:
            x:          (batch_size, num_heads, seq_w, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (batch_size, seq_w, num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_q, depth)

    def call(self, q, k, v, mask):
        """

        Args:
            q:      (batch_size, seq_len_q, d_model)
            k:      (batch_size, seq_len_k, d_model)
            v:      (batch_size, seq_len_k, d_model)
            mask:   (batch_size, q_w, k_w)

        Returns:

        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_k, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_k, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
