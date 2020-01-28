import re
import time
from multiprocessing import Value
from multiprocessing.pool import ThreadPool

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from configuration import Configuration
from mask_loss import MaskLoss
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

    # y_mask_one_hot = tf.one_hot(y_mask, vocab_size)
    token_loss_ps = token_loss_func(y_mask, y_hat_mask)
    token_loss_ps = tf.reduce_sum(y_mask_w * token_loss_ps)
    token_loss_ps = token_loss_ps / (tf.reduce_sum(y_mask_w) + epsilon)

    # sp_one_hot = tf.one_hot(y_sp, 2)
    sp_loss_ps = sp_loss_func(y_sp, y_hat_sp)
    sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=c.b_p_gpu)

    train_loss = token_loss_ps + sp_loss_ps

    return train_loss


def accuracy_function(y_mask, y_mask_w, y_hat_mask, y_sp, y_hat_sp):
    tokens = tf.argmax(y_hat_mask, 2)
    tokens = tf.cast(tokens, tf.float32)
    same_paper = tf.argmax(y_hat_sp, 1)
    same_paper = tf.cast(same_paper, tf.float32)

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


def build_model():
    global c, strategy, model, mask_loss, optimizer, ckpt_manager, token_loss_func, sp_loss_func

    strategy = tf.distribute.MirroredStrategy()

    tokenizer = Tokenizer(c.vocab_folder)
    vocab_size = tokenizer.vocab_size

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

        token_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
        sp_loss_func = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

        optimizer = tf.keras.optimizers.Adam(c.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-6)

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, c.checkpoint_folder, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return


def load_data(data_file, buffer_size):
    global c

    tokenized_sequences = []
    input_masks = []
    segments_ids = []
    masks_indices = []
    masks_weights = []
    tokenized_masks = []
    labels = []

    counter = Value('f', 0)

    with open(data_file, 'r', encoding='utf-8') as stream:
        sequences = stream.readlines()
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

    def load_sample(sample):
        index = -1

        with counter.get_lock():
            counter.value += 1
            index = counter.value

        tokenized_sequence, input_mask, segment_ids, mask_indices, tokenized_mask, mask_weights, label = sample.split(";")

        tokenized_sequence = list(map(float, tokenized_sequence.split(" ")[:-1]))
        assert_sequence_length(len(tokenized_sequence), index)
        tokenized_sequences.append(tokenized_sequence)

        input_mask = list(map(float, input_mask.split(" ")[:-1]))
        assert_sequence_length(len(input_mask), index)
        input_masks.append(input_mask)

        segment_ids = list(map(int, segment_ids.split(" ")[:-1]))
        assert_sequence_length(len(segment_ids), index)
        segments_ids.append(segment_ids)

        mask_indices = list(map(int, mask_indices.split(" ")[:-1]))
        assert_mask_length(len(mask_indices), index)
        masks_indices.append(mask_indices)

        mask_weights = list(map(float, mask_weights.split(" ")[:-1]))
        assert_mask_length(len(mask_weights), index)
        masks_weights.append(mask_weights)

        tokenized_mask = list(map(float, tokenized_mask.split(" ")[:-1]))
        assert_mask_length(len(tokenized_mask), index)
        tokenized_masks.append(tokenized_mask)

        label = float(label)
        labels.append(label)

        printf("DATA LOADING : {0:.3}%", (index / buffer_size) * 1e2)

    pool = ThreadPool(c.cpu_threads)
    pool.map(load_sample, sequences)
    pool.close()
    pool.join()

    print("")
    print("{} has been LOADED".format(data_file))

    tf_tokenized_sequences = tf.data.Dataset.from_tensor_slices(tokenized_sequences)
    tf_tokenized_sequences = tf_tokenized_sequences.batch(c.batch_size)
    tf_input_masks = tf.data.Dataset.from_tensor_slices(input_masks)
    tf_input_masks = tf_input_masks.batch(c.batch_size)
    tf_segments_ids = tf.data.Dataset.from_tensor_slices(segments_ids)
    tf_segments_ids = tf_segments_ids.batch(c.batch_size)
    tf_mask_indices = tf.data.Dataset.from_tensor_slices(masks_indices)
    tf_mask_indices = tf_mask_indices.batch(c.batch_size)
    tf_tokenized_masks = tf.data.Dataset.from_tensor_slices(tokenized_masks)
    tf_tokenized_masks = tf_tokenized_masks.batch(c.batch_size)
    tf_masks_weights = tf.data.Dataset.from_tensor_slices(masks_weights)
    tf_masks_weights = tf_masks_weights.batch(c.batch_size)
    tf_labels = tf.data.Dataset.from_tensor_slices(labels)
    tf_labels = tf_labels.batch(c.batch_size)

    print("{} has been into TF dataset CONVERTED".format(data_file))

    tf_train_dataset = tf.data.Dataset.zip(
            (tf_tokenized_sequences, tf_input_masks, tf_segments_ids, tf_mask_indices, tf_tokenized_masks, tf_masks_weights, tf_labels))

    return tf_train_dataset


def train_model():
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.train:
        return

    tf_train_dataset = load_data(c.train_data_file, c.train_buffer_size)
    tf_test_dataset = load_data(c.test_data_file, c.test_buffer_size)

    with strategy.scope():
        tf_train_dataset = strategy.experimental_distribute_dataset(tf_train_dataset)
        tf_test_dataset = strategy.experimental_distribute_dataset(tf_test_dataset)

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

        template = 'E: {} ({:.2f}%) | Loss: [{:.4f}, {:.4f}] | Mask / Label Acc: [{:.4f}, {:.4f}, {:.4f}, {:.4f}] | delta = {:.2f}'

        tr_steps = c.train_buffer_size / c.batch_size
        va_steps = c.test_buffer_size / c.batch_size + 1e-9

        for e in range(c.epochs):
            start = time.time()

            tr_l1_acc, tr_a1_acc, tr_a2_acc = 0, 0, 0

            for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(tf_train_dataset):
                tr_l1, tr_a1, tr_a2 = distributed_train_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp)
                tr_l1_acc, tr_a1_acc, tr_a2_acc = tr_l1_acc + tr_l1, tr_a1_acc + tr_a1, tr_a2_acc + tr_a2
                printf("TRAIN_STEP : {} ({:.3}%)", batch, ((batch + 1) / tr_steps) * 1e2)

            tr_l1_acc, tr_a1_acc, tr_a2_acc = tr_l1_acc / tr_steps, tr_a1_acc / tr_steps, tr_a2_acc / tr_steps

            va_l1_acc, va_a1_acc, va_a2_acc = 0, 0, 0

            for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(tf_test_dataset):
                va_l1, va_a1, va_a2 = distributed_test_step(in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp)
                va_l1_acc, va_a1_acc, va_a2_acc = va_l1_acc + va_l1, va_a1_acc + va_a1, va_a2_acc + va_a2
                printf("TEST_STEP : {} ({:.3}%)", batch, ((batch + 1) / va_steps) * 1e2)

            va_l1_acc, va_a1_acc, va_a2_acc = va_l1_acc / va_steps, va_a1_acc / va_steps, va_a2_acc / va_steps

            percent = (e / c.epochs) * 1e2
            delta = time.time() - start
            print("\r", end="")
            print(template.format(e, percent, tr_l1_acc, va_l1_acc, tr_a1_acc, tr_a2_acc, va_a1_acc, va_a2_acc, delta))

            if (e + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(e + 1, ckpt_save_path))


def inference():
    global c, model, optimizer, ckpt_manager

    if not c.infer:
        return

    tf_train_dataset = load_data(c.train_data_file, c.train_buffer_size)
    validation = []

    for batch, (in_seq, in_mask, in_seg, in_ind, y_mask, y_weight, y_sp) in enumerate(tf_train_dataset):
        enc_padding_mask = create_mask(in_mask)
        y_hat_mask, y_hat_ns = model(in_seq, enc_padding_mask, in_seg, in_ind)
        err = loss_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_ns)
        mask_accuracy, label_accuracy = accuracy_function(y_mask, y_weight, y_hat_mask, y_sp, y_hat_ns)

        tokens = tf.argmax(y_hat_mask, axis=2)
        same_paper = tf.argmax(y_hat_ns, axis=1)
        mask_len = y_weight.numpy().sum(axis=1)

        y_mask = np.int32(y_mask.numpy())
        tokens = np.int32(tokens.numpy())
        y_sp = np.int32(y_sp.numpy())
        same_paper = same_paper.numpy()
        mask_accuracy = mask_accuracy.numpy()
        label_accuracy = label_accuracy.numpy()

        entry = (y_mask, tokens, mask_len, y_sp, same_paper, mask_accuracy, label_accuracy)
        validation.append(entry)

    validation.sort(key=lambda x: x[5] + x[6], reverse=True)

    with open(c.validation_file, mode='w', encoding='utf-8') as stream:
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
            # for (mask, mask_hat, mask_len, sm, sm_hat) in zip(batch[0], batch[1], batch[2], batch[3], batch[4]):
            #     mask_len = int(mask_len)
            #     for token in mask[0:mask_len]:
            #         stream.write(str(token))
            #         stream.write(" ")
            #     stream.write(";")
            #     for token in mask_hat[0:mask_len]:
            #         stream.write(str(token))
            #         stream.write(" ")
            #     stream.write(";")
            #     stream.write(str(sm))
            #     stream.write(";")
            #     stream.write(str(sm_hat))
            #     stream.write('\n')
        stream.close()


def run_bert():
    build_model()
    train_model()
    inference()
    return


if __name__ == "__main__":
    load_configuration()
    a = time.time()
    run_bert()
    b = (time.time() - a) * 1e3
    print("BERT NETWORK in {0}".format(b))
