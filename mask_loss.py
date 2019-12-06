import tensorflow as tf


class MaskLoss(tf.keras.losses.Loss):

    def __init__(self, batch_size):
        super().__init__(reduction='none')
        self.batch_size = batch_size
        self.token_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        self.sp_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        y_mas, y_wei, y_sp = y_true
        y_hat_mas, y_hat_sp = y_pred
        vocab_size = y_hat_mas.shape[2]
        epsilon = 1e-5

        y_mas_one_hot = tf.one_hot(y_mas, vocab_size)
        token_loss_ps = self.token_loss_func(y_mas_one_hot, y_hat_mas)
        token_loss_ps = tf.reduce_sum(y_wei * token_loss_ps)
        token_loss_ps = token_loss_ps / (tf.reduce_sum(y_wei) + epsilon)

        sp_one_hot = tf.one_hot(y_sp, 2)
        sp_loss_ps = self.sp_loss_func(sp_one_hot, y_hat_sp)
        sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=self.batch_size)

        train_loss = token_loss_ps + sp_loss_ps

        tokens = tf.argmax(y_hat_mas, 2)
        same_paper = tf.argmax(y_hat_sp, 1)

        y_wei = tf.cast(y_wei, tf.int64)
        y_mas = tf.cast(y_mas, tf.int64)
        y_sp = tf.cast(y_sp, tf.int64)
        mask_accuracy = tf.abs(tokens * y_wei - y_mas)
        mask_accuracy = 1 - tf.reduce_sum(tf.cast(mask_accuracy > 0, dtype=tf.int64)) / tf.reduce_sum(y_wei)
        label_accuracy = 1 - tf.reduce_sum(tf.abs(same_paper - y_sp)) / self.batch_size

        mask_accuracy = tf.cast(mask_accuracy, tf.float32)
        label_accuracy = tf.cast(label_accuracy, tf.float32)

        return train_loss, mask_accuracy, label_accuracy
