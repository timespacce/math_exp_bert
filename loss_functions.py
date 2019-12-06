import tensorflow as tf


class MaskLoss(tf.keras.losses.Loss):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.token_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        y_mask, y_mask_w = tf.cast(y_true[:, 0, :], dtype=tf.int32), y_true[:, 1, :]
        y_hat_mask = y_pred
        vocab_size = y_hat_mask.shape[2]
        epsilon = 1e-5

        y_mask_one_hot = tf.one_hot(y_mask, vocab_size)
        token_loss_ps = self.token_loss_func(y_mask_one_hot, y_hat_mask)
        token_loss_ps = tf.reduce_sum(y_mask_w * token_loss_ps)
        token_loss_ps = token_loss_ps / (tf.reduce_sum(y_mask_w) + epsilon)

        return token_loss_ps


class SamePaperLoss(tf.keras.losses.Loss):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.ns_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        label = y_true
        y_hat_ns = y_pred

        ns_one_hot = tf.one_hot(label, 2)
        ns_loss_ps = self.ns_loss_func(ns_one_hot, y_hat_ns)
        ns_loss_ps = tf.nn.compute_average_loss(ns_loss_ps, global_batch_size=self.batch_size)

        return ns_loss_ps
