from configuration import Configuration
import time
import numpy as np
import os
import tensorflow as tf


class DataLoader:
    c = None

    def __init__(self, configuration: Configuration):
        self.c = configuration
        return

    def load_data_block(self, target, block_id, buffer_size):
        if self.c.pre_training:
            return self.load_pre_training_block(target=target, block_id=block_id, buffer_size=buffer_size)

        if self.c.fine_tuning == 'GENERATIVE':
            return self.load_fine_tuning_derivation_block(target=target, block_id=block_id, buffer_size=buffer_size)

        if self.c.fine_tuning == 'DISCRIMINATIVE':
            return self.load_fine_tuning_derivation_block(target=target, block_id=block_id, buffer_size=buffer_size)

        if self.c.fine_tuning == 'EQUALITY':
            return self.load_fine_tuning_equality_block(target=target, block_id=block_id, buffer_size=buffer_size)

        return None

    def load_pre_training_block(self, target, block_id, buffer_size):
        begin = time.time()

        prefix = target + "block_{0}/".format(block_id)
        x_file = prefix + self.c.x_file
        x_id_file = prefix + self.c.x_id_file
        x_seg_file = prefix + self.c.x_seg_file
        y_mask_file = prefix + self.c.y_mask_file
        y_id_file = prefix + self.c.y_id_file
        y_w_file = prefix + self.c.y_w_file
        sp_file = prefix + self.c.sp_file

        size = os.path.getsize(sp_file)
        count = size // 4
        count = np.minimum(count, buffer_size)
        count = (count // self.c.batch_size) * self.c.batch_size

        xs_s = open(x_file, 'rb')
        xs_id_s = open(x_id_file, 'rb')
        xs_seg_s = open(x_seg_file, 'rb')
        ys_mask_s = open(y_mask_file, 'rb')
        ys_id_s = open(y_id_file, 'rb')
        ys_w_s = open(y_w_file, 'rb')
        sps_s = open(sp_file, 'rb')

        b = xs_s.read()
        b_s = self.c.max_seq_len * count
        xs = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
        xs = xs.reshape((-1, self.c.max_seq_len))
        xs_s.close()

        b = xs_id_s.read()
        b_s = self.c.max_seq_len * count
        xs_id = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
        xs_id = xs_id.reshape((-1, self.c.max_seq_len))
        xs_id_s.close()

        b = xs_seg_s.read()
        b_s = self.c.max_seq_len * count
        xs_seg = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        xs_seg = xs_seg.reshape((-1, self.c.max_seq_len))
        xs_seg_s.close()

        b = ys_mask_s.read()
        b_s = self.c.max_mask_len * count
        ys_mask = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        ys_mask = ys_mask.reshape((-1, self.c.max_mask_len))
        ys_mask_s.close()

        b = ys_id_s.read()
        b_s = self.c.max_mask_len * count
        ys_id = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        ys_id = ys_id.reshape((-1, self.c.max_mask_len))
        ys_id_s.close()

        b = ys_w_s.read()
        b_s = self.c.max_mask_len * count
        ys_w = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
        ys_w = ys_w.reshape((-1, self.c.max_mask_len))
        ys_w_s.close()

        b = sps_s.read()
        b_s = count
        sps = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        sps = sps.reshape((-1,))
        sps_s.close()

        if xs.shape[0] != xs_id.shape[0] != xs_seg.shape[0]:
            print("XS LEN is not correct.")
            exit()

        if ys_mask.shape[0] != ys_id.shape[0] != ys_w.shape[0]:
            print("YS LEN is not correct.")
            exit()

        if sps.shape[0] != count:
            print("SP LEN is not correct.")
            exit()

        tf_xs = tf.data.Dataset.from_tensor_slices(xs)
        tf_xs_id = tf.data.Dataset.from_tensor_slices(xs_id)
        tf_xs_seg = tf.data.Dataset.from_tensor_slices(xs_seg)
        tf_ys_mask = tf.data.Dataset.from_tensor_slices(ys_mask)
        tf_ys_id = tf.data.Dataset.from_tensor_slices(ys_id)
        tf_ys_w = tf.data.Dataset.from_tensor_slices(ys_w)
        tf_sps = tf.data.Dataset.from_tensor_slices(sps)

        tf_dataset = tf.data.Dataset.zip((tf_xs, tf_xs_id, tf_xs_seg, tf_ys_mask, tf_ys_id, tf_ys_w, tf_sps))
        tf_dataset = tf_dataset.shuffle(buffer_size=count, reshuffle_each_iteration=True)
        tf_dataset = tf_dataset.batch(batch_size=self.c.batch_size)

        runtime = time.time() - begin

        if self.c.debug:
            print("BLOCK_{0} with {1} in {2:.3} s.".format(block_id, count, runtime))

        return tf_dataset

    def load_fine_tuning_derivation_block(self, target, block_id, buffer_size):
        begin = time.time()

        prefix = target + "block_{0}/".format(block_id)
        x_file = prefix + self.c.x_file
        x_id_file = prefix + self.c.x_id_file
        x_seg_file = prefix + self.c.x_seg_file
        y_file = prefix + self.c.y_file

        size = os.path.getsize(x_file)
        count = size // (4 * self.c.max_seq_len)
        count = np.minimum(count, buffer_size)
        count = (count // self.c.batch_size) * self.c.batch_size

        xs_s = open(x_file, 'rb')
        xs_id_s = open(x_id_file, 'rb')
        xs_seg_s = open(x_seg_file, 'rb')
        ys_s = open(y_file, 'rb')

        b = xs_s.read()
        b_s = self.c.max_seq_len * count
        xs = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
        xs = xs.reshape((-1, self.c.max_seq_len))
        xs_s.close()

        b = xs_id_s.read()
        b_s = self.c.max_seq_len * count
        xs_id = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
        xs_id = xs_id.reshape((-1, self.c.max_seq_len))
        xs_id_s.close()

        b = xs_seg_s.read()
        b_s = self.c.max_seq_len * count
        xs_seg = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        xs_seg = xs_seg.reshape((-1, self.c.max_seq_len))
        xs_seg_s.close()

        b = ys_s.read()
        b_s = self.c.max_seq_len * count if self.c.fine_tuning == 'GENERATIVE' else 1 * count
        ys_shape = (-1, self.c.max_seq_len) if self.c.fine_tuning == 'GENERATIVE' else (-1, 1)
        ys = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
        ys = ys.reshape(ys_shape)
        ys_s.close()

        if xs.shape[0] != xs_id.shape[0] != xs_seg.shape[0]:
            print("XS LEN is not correct.")
            exit()

        if ys.shape[0] != count:
            print("DC LEN is not correct.")
            exit()

        tf_xs = tf.data.Dataset.from_tensor_slices(xs)
        tf_xs_id = tf.data.Dataset.from_tensor_slices(xs_id)
        tf_xs_seg = tf.data.Dataset.from_tensor_slices(xs_seg)
        tf_ys = tf.data.Dataset.from_tensor_slices(ys)

        tf_dataset = tf.data.Dataset.zip((tf_xs, tf_xs_id, tf_xs_seg, tf_ys))
        tf_dataset = tf_dataset.shuffle(buffer_size=count, reshuffle_each_iteration=True)
        tf_dataset = tf_dataset.batch(batch_size=self.c.batch_size)

        runtime = time.time() - begin

        if self.c.debug:
            print("BLOCK_{0} with {1} in {2:.3} s.".format(block_id, count, runtime))

        return tf_dataset

    def load_fine_tuning_equality_block(self, target, block_id, buffer_size):
        begin = time.time()

        prefix = target + "block_{0}/".format(block_id)
        xl_file, xr_file = prefix + "xls.data", prefix + "xrs.data"
        xl_id_file, xr_id_file = prefix + "xls_id.data", prefix + "xrs_id.data"
        xl_seg_file, xr_seg_file = prefix + "xls_seg.data", prefix + "xrs_seg.data"

        size = os.path.getsize(xl_file)
        count = size // (4 * self.c.max_seq_len)
        count = np.minimum(count, buffer_size)
        count = (count // self.c.batch_size) * self.c.batch_size

        xl_s, xr_s = open(xl_file, 'rb'), open(xr_file, 'rb')
        xl_id_s, xr_id_s = open(xl_id_file, 'rb'), open(xr_id_file, 'rb')
        xl_seg_s, xr_seg_s = open(xl_seg_file, 'rb'), open(xr_seg_file, 'rb')

        xls = self.load_from_buffer(xl_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.float32)
        xrs = self.load_from_buffer(xr_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.float32)
        xls_id = self.load_from_buffer(xl_id_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.float32)
        xrs_id = self.load_from_buffer(xr_id_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.float32)
        xls_seg = self.load_from_buffer(xl_seg_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.int32)
        xrs_seg = self.load_from_buffer(xr_seg_s, self.c.max_seq_len * count, (-1, self.c.max_seq_len), np.int32)

        ##

        randomize = np.arange(count)
        np.random.shuffle(randomize)
        xls = xls[randomize]
        xrs = xrs[randomize]
        xls_id = xls_id[randomize]
        xrs_id = xrs_id[randomize]
        xls_seg = xls_seg[randomize]
        xrs_seg = xrs_seg[randomize]

        ##

        xlrs = np.insert(xrs, np.arange(xls.shape[0]), xls, axis=0)
        xlrs = xlrs.reshape((-1, self.c.max_seq_len))
        xlrs_id = np.insert(xrs_id, np.arange(xls_id.shape[0]), xls_id, axis=0)
        xlrs_id = xlrs_id.reshape((-1, self.c.max_seq_len))
        xlrs_seg = np.insert(xrs_seg, np.arange(xls_seg.shape[0]), xls_seg, axis=0)
        xlrs_seg = xlrs_seg.reshape((-1, self.c.max_seq_len))

        if xls.shape[0] != xrs.shape[0] or xls.shape[1] != xrs.shape[1]:
            print("XLS / XRS LEN is not correct.")
            exit()

        tf_xlrs = tf.data.Dataset.from_tensor_slices(xlrs)
        tf_xlrs_id = tf.data.Dataset.from_tensor_slices(xlrs_id)
        tf_xlrs_seg = tf.data.Dataset.from_tensor_slices(xlrs_seg)
        tf_dataset = tf.data.Dataset.zip((tf_xlrs, tf_xlrs_id, tf_xlrs_seg))
        tf_dataset = tf_dataset.batch(batch_size=self.c.batch_size)

        runtime = time.time() - begin

        if self.c.debug:
            print("BLOCK_{0} with {1} in {2:.3} s.".format(block_id, count, runtime))

        return tf_dataset

    def load_from_buffer(self, buffer, count, shape, data_type):
        b = buffer.read()
        data = np.frombuffer(buffer=b, count=count, dtype=data_type)
        data = data.reshape(shape)
        buffer.close()
        return data
