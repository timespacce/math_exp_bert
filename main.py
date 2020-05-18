import re
import time
from multiprocessing import Value
from multiprocessing.pool import ThreadPool

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from configuration import Configuration
from mask_loss import MaskLoss
from optimization import LearningRateScheduler, CustomOptimizer
from tokenizer import Tokenizer
from transformer import Transformer

c = None

model = None
optimizer = None
ckpt_manager = None

strategy = None

token_loss_func = None
sp_loss_func = None
mask_loss = None


def printf(template, *args):
    print("\r", end="")
    print(template.format(*args), end="", flush=True)


def create_mask(input_mask):
    batch_size = input_mask.shape[0]
    seq_len = input_mask.shape[1]
    broadcast = tf.ones((batch_size, seq_len, 1), dtype=np.float32)
    mask = tf.reshape(input_mask, shape=(batch_size, 1, seq_len))
    enc_padding_mask = broadcast * mask  # (batch_size, seq_len, seq_len)
    return enc_padding_mask


def loss_function(y_mask, y_mask_w, y_hat_mask, y_sp, y_hat_sp):
    global c, token_loss_func, sp_loss_func

    epsilon = 1e-5

    tokens = tf.one_hot(y_mask, c.vocab_size)
    tokens = tf.cast(tokens, tf.float32)
    token_loss_ps = token_loss_func(tokens, y_hat_mask)
    token_loss_ps = tf.reduce_sum(y_mask_w * token_loss_ps)
    token_loss_ps = token_loss_ps / (tf.reduce_sum(y_mask_w) + epsilon)

    sp = tf.one_hot(y_sp, 2)
    sp = tf.cast(sp, tf.float32)
    sp_loss_ps = sp_loss_func(sp, y_hat_sp)
    sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=c.b_p_gpu)

    train_loss = token_loss_ps + sp_loss_ps

    return train_loss


def accuracy_function(y_mask, y_mask_w, y_hat_mask, y_sp, y_hat_sp):
    tokens = tf.argmax(y_hat_mask, 2)
    tokens = tf.cast(tokens, tf.float32)
    same_paper = tf.argmax(y_hat_sp, 1)
    same_paper = tf.cast(same_paper, tf.float32)

    y_mask = tf.cast(y_mask, tf.float32)
    y_sp = tf.cast(y_sp, tf.float32)
    mask_accuracy = tf.abs(tokens * y_mask_w - y_mask)
    mask_accuracy = mask_accuracy / (mask_accuracy + 1e-15)
    mask_accuracy = tf.reduce_sum(mask_accuracy, axis=1) / tf.reduce_sum(y_mask_w, axis=1)
    mask_accuracy = 1 - tf.reduce_mean(mask_accuracy)
    sp_accuracy = 1 - tf.reduce_mean(tf.abs(same_paper - y_sp))

    return mask_accuracy, sp_accuracy


def load_configuration():
    global c

    configuration_file = "configuration.json"
    c = Configuration(configuration_file)

    return


def build_model():
    global c, strategy, model, mask_loss, optimizer, ckpt_manager, token_loss_func, sp_loss_func

    strategy = tf.distribute.MirroredStrategy()

    steps = c.epochs * (c.train_buffer_size / c.batch_size)

    with strategy.scope():
        model = Transformer(max_seq_len=c.max_seq_len,
                            b_p_gpu=c.b_p_gpu,
                            num_layers=c.num_layers,
                            hidden_size=c.hidden_size,
                            intermediate_size=c.intermediate_size,
                            num_heads=c.num_heads,
                            input_vocab_size=c.vocab_size,
                            target_vocab_size=2,
                            rate=c.drop_rate)

        token_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')
        sp_loss_func = tf.keras.losses.CategoricalCrossentropy(reduction='none')

        alpha_2, warmup_steps, decay_steps, power = 0.0, steps // 2, steps, 1

        learning_rate_scheduler = LearningRateScheduler(alpha_1=c.learning_rate,
                                                        alpha_2=alpha_2,
                                                        hidden_size=c.hidden_size,
                                                        training_steps=steps,
                                                        warmup_steps=warmup_steps,
                                                        decay_steps=decay_steps,
                                                        power=power)

        # optimizer = tf.keras.optimizers.Adam(learning_rate_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-6)
        optimizer = CustomOptimizer(learning_rate=learning_rate_scheduler, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.01)

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, c.checkpoint_folder, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return


def load_data(data_file, buffer_size):
    global c

    if buffer_size <= 0:
        return []

    counter = Value('i', -1)
    mutex = Value('i', 0)

    with open(data_file, 'r', encoding='utf-8') as stream:
        sequences = stream.readlines()
        sequences_len = len(sequences)
        print("{} with {} sequences".format(data_file, sequences_len))
        stream.close()

    sequences = sequences[:buffer_size]

    def assert_sequence_length(width, index):
        if width != c.max_seq_len:
            print("ERROR at {} got {} != is {}".format(index, width, c.max_seq_len))
            exit(1)

    def assert_mask_length(width, index):
        if width != c.max_mask_len:
            print("ERROR at {} got {} != is {}".format(index, width, c.max_mask_len))
            exit(1)

    in_seqs = np.zeros(shape=(buffer_size, c.max_seq_len), dtype=np.float32)
    in_masks = np.zeros(shape=(buffer_size, c.max_seq_len), dtype=np.float32)
    in_segs = np.zeros(shape=(buffer_size, c.max_seq_len), dtype=np.int32)
    mask_inds = np.zeros(shape=(buffer_size, c.max_mask_len), dtype=np.int32)
    y_masks = np.zeros(shape=(buffer_size, c.max_mask_len), dtype=np.int32)
    mask_ws = np.zeros(shape=(buffer_size, c.max_mask_len), dtype=np.float32)
    sps = np.zeros(shape=buffer_size, dtype=np.int32)

    def load_sample(sample):
        with counter.get_lock():
            counter.value += 1
            index = counter.value

        in_seq, in_mask, in_seg, mask_ind, y_mask, mask_w, sp = sample.split(";")

        in_seq = np.fromstring(in_seq, dtype=np.float32, sep=' ')
        assert_sequence_length(len(in_seq), index)

        in_mask = np.fromstring(in_mask, dtype=np.float32, sep=' ')
        assert_sequence_length(len(in_mask), index)

        in_seg = np.fromstring(in_seg, dtype=np.int32, sep=' ')
        assert_sequence_length(len(in_seg), index)

        mask_ind = np.fromstring(mask_ind, dtype=np.int32, sep=' ')
        assert_mask_length(len(mask_ind), index)

        mask_w = np.fromstring(mask_w, dtype=np.float32, sep=' ')
        assert_mask_length(len(mask_w), index)

        y_mask = np.fromstring(y_mask, dtype=np.int32, sep=' ')
        assert_mask_length(len(y_mask), index)

        sp = float(sp)

        with mutex.get_lock():
            in_seqs[index] = in_seq
            in_masks[index] = in_mask
            in_segs[index] = in_seg
            mask_inds[index] = mask_ind
            y_masks[index] = y_mask
            mask_ws[index] = mask_w
            sps[index] = sp

        printf("DATA LOADING : {0:.3}% ", (index / buffer_size) * 1e2)

    pool = ThreadPool(c.cpu_threads)
    pool.map(load_sample, sequences)
    pool.close()
    pool.join()

    print("{} has been LOADED".format(data_file))

    in_seqs = tf.data.Dataset.from_tensor_slices(in_seqs)
    in_masks = tf.data.Dataset.from_tensor_slices(in_masks)
    in_segs = tf.data.Dataset.from_tensor_slices(in_segs)
    mask_inds = tf.data.Dataset.from_tensor_slices(mask_inds)
    y_masks = tf.data.Dataset.from_tensor_slices(y_masks)
    mask_ws = tf.data.Dataset.from_tensor_slices(mask_ws)
    sps = tf.data.Dataset.from_tensor_slices(sps)

    train_data = tf.data.Dataset.zip((in_seqs, in_masks, in_segs, mask_inds, y_masks, mask_ws, sps))
    train_data = train_data.batch(c.batch_size)

    print("{} has been into TF dataset CONVERTED".format(data_file))

    return train_data


def train_model(train_data, test_data):
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.train or c.train_buffer_size is 0:
        return

    with strategy.scope():
        tf_train_dataset = strategy.experimental_distribute_dataset(train_data)
        tf_test_dataset = strategy.experimental_distribute_dataset(test_data)

        def train_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp):
            enc_padding_mask = create_mask(in_mask)

            with tf.GradientTape() as tape:
                y_hat_mask, y_hat_sp = model(in_seq, enc_padding_mask, in_seg, in_ind)
                loss = loss_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_sp)
                mask_accuracy, label_accuracy = accuracy_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_sp)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss, mask_accuracy, label_accuracy

        def test_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp):
            enc_padding_mask = create_mask(in_mask)

            with tf.GradientTape() as tape:
                y_hat_mask, y_hat_sp = model(in_seq, enc_padding_mask, in_seg, in_ind)
                loss = loss_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_sp)
                mask_accuracy, label_accuracy = accuracy_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_sp)

            return loss, mask_accuracy, label_accuracy

        @tf.function
        def distributed_train_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp):
            loss, mask_accuracy, label_accuracy = strategy.experimental_run_v2(train_step, args=(in_seq,
                                                                                                 in_mask,
                                                                                                 in_seg,
                                                                                                 in_ind,
                                                                                                 y_mask,
                                                                                                 y_weight,
                                                                                                 y_sp))
            if strategy.num_replicas_in_sync > 1:
                return tf.reduce_sum(loss.values, axis=-1) / c.gpu_count, \
                       tf.reduce_sum(mask_accuracy.values, axis=-1) / c.gpu_count, \
                       tf.reduce_sum(label_accuracy.values, axis=-1) / c.gpu_count
            else:
                return loss, mask_accuracy, label_accuracy

        @tf.function
        def distributed_test_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp):
            loss, mask_accuracy, label_accuracy = strategy.experimental_run_v2(test_step, args=(in_seq,
                                                                                                in_mask,
                                                                                                in_seg,
                                                                                                in_ind,
                                                                                                y_mask,
                                                                                                y_weight,
                                                                                                y_sp))
            if strategy.num_replicas_in_sync > 1:
                return tf.reduce_sum(loss.values, axis=-1) / c.gpu_count, \
                       tf.reduce_sum(mask_accuracy.values, axis=-1) / c.gpu_count, \
                       tf.reduce_sum(label_accuracy.values, axis=-1) / c.gpu_count
            else:
                return loss, mask_accuracy, label_accuracy

        template = 'E: {} ({:.2f}%) | Loss: [{:.4f}, {:.4f}] | Mask / Label Acc: [{:.4f}, {:.4f}, {:.4f}, {:.4f}] | delta = {:.2f} \n'

        tr_steps = c.train_buffer_size / c.batch_size
        va_steps = c.test_buffer_size / c.batch_size + 1e-9

        for e in range(c.epochs):
            start = time.time()

            tr_l1_acc, tr_a1_acc, tr_a2_acc = 0, 0, 0

            for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(tf_train_dataset):
                tr_l1, tr_a1, tr_a2 = distributed_train_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp)

                w = batch + 1
                tr_l1_acc, tr_a1_acc, tr_a2_acc = tr_l1_acc + tr_l1, tr_a1_acc + tr_a1, tr_a2_acc + tr_a2
                tr_l1_mu, tr_a1_mu, tr_a2_mu = tr_l1_acc / w, tr_a1_acc / w, tr_a2_acc / w
                percent = 1e2 * w / tr_steps
                printf("TRAIN_STEP : {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", batch, percent, tr_l1_mu, tr_a1_mu, tr_a2_mu)

            tr_l1_acc, tr_a1_acc, tr_a2_acc = tr_l1_acc / tr_steps, tr_a1_acc / tr_steps, tr_a2_acc / tr_steps

            va_l1_acc, va_a1_acc, va_a2_acc = 0, 0, 0

            for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(tf_test_dataset):
                va_l1, va_a1, va_a2 = distributed_test_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp)

                w = batch + 1
                va_l1_acc, va_a1_acc, va_a2_acc = va_l1_acc + va_l1, va_a1_acc + va_a1, va_a2_acc + va_a2
                va_l1_mu, va_a1_mu, va_a2_mu = va_l1_acc / w, va_a1_acc / w, va_a2_acc / w
                percent = 1e2 * w / va_steps
                printf("TEST_STEP : {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", batch, percent, va_l1_mu, va_a1_mu, va_a2_mu)

            va_l1_acc, va_a1_acc, va_a2_acc = va_l1_acc / va_steps, va_a1_acc / va_steps, va_a2_acc / va_steps

            percent = (e / c.epochs) * 1e2
            delta = time.time() - start
            printf(template.format(e, percent, tr_l1_acc, va_l1_acc, tr_a1_acc, tr_a2_acc, va_a1_acc, va_a2_acc, delta))

            if (e + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(e + 1, ckpt_save_path))


def inference(dataset, validation_file, buffer_size):
    global c, model, optimizer, ckpt_manager

    if not c.infer or buffer_size is 0:
        return

    validation = []

    steps = buffer_size / c.batch_size
    l1_acc, a1_acc, a2_acc = 0, 0, 0

    for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(dataset):
        enc_padding_mask = create_mask(in_mask)
        y_hat_mask, y_hat_ns = model(in_seq, enc_padding_mask, in_seg, in_ind)
        l1 = loss_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_ns)
        a1, a2 = accuracy_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_ns)
        l1_acc, a1_acc, a2_acc = l1_acc + l1, a1_acc + a1, a2_acc + a2

        tokens = tf.argmax(y_hat_mask, axis=2)
        same_paper = tf.argmax(y_hat_ns, axis=1)
        mask_len = y_weight.numpy().sum(axis=1)

        y_mask = np.int32(y_mask.numpy())
        tokens = np.int32(tokens.numpy())
        y_sp = np.int32(y_sp.numpy())
        same_paper = same_paper.numpy()
        mask_accuracy = a1.numpy()
        label_accuracy = a2.numpy()

        entry = (y_mask, tokens, mask_len, y_sp, same_paper, mask_accuracy, label_accuracy)
        validation.append(entry)

        w = batch + 1
        l1_mu, a1_mu, a2_mu = l1_acc / w, a1_acc / w, a2_acc / w
        percent = 1e2 * (batch + 1) / steps
        printf("INFERENCE : {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", batch, percent, l1_mu, a1_mu, a2_mu)

    l1_acc, a1_acc, a2_acc = l1_acc / steps, a1_acc / steps, a2_acc / steps

    template = "VALIDATION : Loss = {:.4} Mask / Label = {:.4} {:.4}"
    print(template.format(l1_acc, a1_acc, a2_acc))

    validation.sort(key=lambda x: x[5] + x[6], reverse=True)

    with open(validation_file, mode='w', encoding='utf-8') as stream:
        stream.write("L1: ")
        stream.write(str(l1_acc.numpy()))
        stream.write(" - A1: ")
        stream.write(str(a1_acc.numpy()))
        stream.write(" - A2: ")
        stream.write(str(a2_acc.numpy()))
        stream.write('\n')
        for index, batch in enumerate(validation):
            mask_accuracy = batch[5]
            label_accuracy = batch[6]
            stream.write(str(index))
            stream.write(" - ")
            stream.write(str(round(mask_accuracy, 5)))
            stream.write(" - ")
            stream.write(str(round(label_accuracy, 5)))
            stream.write(" - ")
            stream.write('\n')
            if c.debug:
                for (mask, mask_hat, mask_len, sm, sm_hat) in zip(batch[0], batch[1], batch[2], batch[3], batch[4]):
                    mask_len = int(mask_len)
                    for token in mask[0:mask_len]:
                        stream.write(str(token))
                        stream.write(" ")
                    stream.write(";")
                    for token in mask_hat[0:mask_len]:
                        stream.write(str(token))
                        stream.write(" ")
                    stream.write(";")
                    stream.write(str(sm))
                    stream.write(";")
                    stream.write(str(sm_hat))
                    stream.write('\n')
        stream.close()


def run_bert():
    global c

    load_configuration()
    build_model()
    train_data = load_data(c.train_data_file, c.train_buffer_size)
    test_data = load_data(c.test_data_file, c.test_buffer_size)
    train_model(train_data, test_data)
    inference(train_data, c.train_validation_file, c.train_buffer_size)
    inference(test_data, c.test_validation_file, c.test_buffer_size)
    return


if __name__ == "__main__":
    a = time.time()
    run_bert()
    b = (time.time() - a)
    print("BERT NETWORK in {0} s".format(b))
