import tensorflow as tf


class MaskLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size):
        super().__init__(reduction='none')
        self.batch_size = batch_size
        self.token_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.sp_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        y_mask, y_weight, sp = y_true
        y_hat_mask, y_hat_sp = y_pred

        vocab_size = y_hat_mask.shape[2]
        epsilon = 1e-5

        y_mask_one_hot = tf.one_hot(y_mask, vocab_size)
        token_loss_ps = self.token_loss_func(y_mask_one_hot, y_hat_mask)
        token_loss_ps = tf.reduce_sum(y_weight * token_loss_ps)
        token_loss_ps = token_loss_ps / (tf.reduce_sum(y_weight) + epsilon)

        ns_one_hot = tf.one_hot(sp, 2)
        sp_loss_ps = self.sp_loss_func(ns_one_hot, y_hat_sp)
        sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=self.batch_size)

        train_loss = token_loss_ps + sp_loss_ps
        return train_loss


class L1(tf.keras.losses.Loss):
    def __init__(self, batch_size, b_p_gpu, pred_len):
        super().__init__(reduction='none')
        self.batch_size = batch_size
        self.b_p_gpu = b_p_gpu
        self.pred_len = pred_len
        self.token_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        y_mask = tf.cast(y_true[:, :self.pred_len], tf.int32)
        y_weight = y_true[:, self.pred_len:]
        y_hat_mask = y_pred

        vocab_size = y_hat_mask.shape[2]
        epsilon = 1e-5

        y_mask_one_hot = tf.one_hot(y_mask, vocab_size)
        token_loss_ps = self.token_loss_func(y_mask_one_hot, y_hat_mask)
        token_loss_ps = tf.reduce_sum(y_weight * token_loss_ps)
        token_loss_ps = token_loss_ps / (tf.reduce_sum(y_weight) + epsilon)

        return token_loss_ps


class L2(tf.keras.losses.Loss):
    def __init__(self, batch_size, b_p_gpu):
        super().__init__(reduction='none')
        self.batch_size = batch_size
        self.b_p_gpu = b_p_gpu
        self.sp_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        sp = y_true
        y_hat_sp = y_pred

        sp = tf.reshape(sp, shape=(self.b_p_gpu,))
        sp = tf.cast(sp, dtype=tf.int64)
        ns_one_hot = tf.one_hot(sp, 2)
        sp_loss_ps = self.sp_loss_func(ns_one_hot, y_hat_sp)
        sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=self.batch_size)

        return sp_loss_ps
