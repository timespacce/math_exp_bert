import unittest

import tensorflow as tf
import numpy as np

from mha import scaled_dot_product_attention, MultiHeadAttention


class ScaledDotSelfAttentionTest(unittest.TestCase):

    def test_scaled_dot_self_attention(self):
        test_case = tf.test.TestCase()

        k = tf.constant([[10, 0, 0],
                         [0, 10, 0],
                         [0, 0, 10],
                         [0, 0, 10]], dtype=tf.float32)

        v = tf.constant([[1, 0],
                         [10, 0],
                         [100, 5],
                         [1000, 6]], dtype=tf.float32)

        # This `query` aligns with the second `key`, so the second `value` is returned.
        q = tf.constant([[0, 10, 0]], dtype=tf.float32)

        output, attention_weights = scaled_dot_product_attention(q, k, v)

        test_case.assertAllClose(output, tf.constant([[10, 0]], dtype=np.float32))
        test_case.assertAllClose(attention_weights, tf.constant([[0, 1, 0, 0]], dtype=np.float32))

        # # This query aligns with a repeated key (third and fourth), so all associated values get averaged.
        q = tf.constant([[0, 0, 10]], dtype=tf.float32)

        output, attention_weights = scaled_dot_product_attention(q, k, v)

        test_case.assertAllClose(output, tf.constant([[550, 5.5]], dtype=np.float32))
        test_case.assertAllClose(attention_weights, tf.constant([[0, 0, 0.5, 0.5]], dtype=np.float32))

        self.assertTrue(True)

        return


class MultiHeadAttentionTest(unittest.TestCase):

    def test_multi_head_attention(self):
        test_case = tf.test.TestCase()
        d_model = 128
        num_heads = 8

        mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        k = tf.constant([[10, 0, 0],
                         [0, 10, 0],
                         [0, 0, 10],
                         [0, 0, 10]], dtype=tf.float32)

        v = tf.constant([[1, 0],
                         [10, 0],
                         [100, 5],
                         [1000, 6]], dtype=tf.float32)

        # This `query` aligns with the second `key`, so the second `value` is returned.
        q = tf.constant([[0, 10, 0]], dtype=tf.float32)

        mask = None

        output, attention_weights = mha(q, k, v, mask)

        self.assertTrue(True)

        return


if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)

    unittest.main()
