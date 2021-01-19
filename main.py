import os
import sys
import time
import numpy as np
import tensorflow as tf
from configuration import Configuration
from optimization import LearningRateScheduler, CustomOptimizer
from transformer import Transformer

c = None

model = None
optimizer = None
ckpt_manager = None

strategy = None

token_loss_func = None
sp_loss_func = None
mask_loss = None

#

mode = 'DISCRIMINATIVE'


def printf(template, *args):
    print("\r", end="")
    print(template.format(*args), end="", flush=True)


def create_mask(input_mask):
    global c

    batch_size = input_mask.shape[0]
    broadcast = tf.ones((batch_size, c.max_seq_len, 1), dtype=np.float32)
    mask = tf.reshape(input_mask, shape=(batch_size, 1, c.max_seq_len))
    enc_padding_mask = broadcast * mask  # (batch_size, seq_len, seq_len)
    return enc_padding_mask


def loss_function(y_mask, y_mask_w, y_hat_mask, y_sp, y_hat_sp):
    global c, token_loss_func, sp_loss_func

    epsilon = 1e-5

    tokens = tf.one_hot(y_mask, c.vocab_size, dtype=tf.float32)
    token_loss_ps = token_loss_func(tokens, y_hat_mask)
    token_loss_ps = tf.reduce_sum(y_mask_w * token_loss_ps)
    token_loss_ps = token_loss_ps / (tf.reduce_sum(y_mask_w) + epsilon)

    sp = tf.one_hot(y_sp, 2, dtype=tf.float32)
    sp_loss_ps = sp_loss_func(sp, y_hat_sp)
    sp_loss_ps = tf.nn.compute_average_loss(sp_loss_ps, global_batch_size=c.b_p_gpu)

    train_loss = token_loss_ps + sp_loss_ps

    return train_loss


def accuracy_function(y_mask, y_mask_w, y_hat_mask, y_sp, y_hat_sp):
    tokens = tf.cast(tf.argmax(y_hat_mask, 2), dtype=tf.float32)
    same_paper = tf.cast(tf.argmax(y_hat_sp, 1), dtype=tf.float32)

    y_mask = tf.cast(y_mask, tf.float32)
    y_sp = tf.cast(y_sp, tf.float32)
    mask_error = tf.abs(tokens * y_mask_w - y_mask)
    mask_error = mask_error / (mask_error + 1e-15)
    mask_error = tf.reduce_sum(mask_error, axis=1) / tf.reduce_sum(y_mask_w, axis=1)
    mask_accuracy = 1 - tf.reduce_mean(mask_error)
    sp_accuracy = 1 - tf.reduce_mean(tf.abs(same_paper - y_sp))

    return mask_accuracy, sp_accuracy


def load_configuration():
    global c
    if len(sys.argv) < 2:
        print("CONFIGURATION-FILE is expected as first argument.")
        exit(1)
    configuration_file = sys.argv[1]
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

        if c.freeze:
            for layer in model.layers[:c.freeze_interval]:
                layer.trainable = True

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
    if ckpt_manager.latest_checkpoint and c.reload:
        checkpoint_index = -1
        status = ckpt.restore(ckpt_manager.checkpoints[checkpoint_index])
        status.expect_partial()
        print('CHECKPOINT INDEX = {0} restored.'.format(checkpoint_index))

    return


def load_pre_training_block(target, block_id, buffer_size):
    global c

    begin = time.time()

    prefix = target + "block_{0}/".format(block_id)
    x_file = prefix + c.x_file
    x_id_file = prefix + c.x_id_file
    x_seg_file = prefix + c.x_seg_file
    y_mask_file = prefix + c.y_mask_file
    y_id_file = prefix + c.y_id_file
    y_w_file = prefix + c.y_w_file
    sp_file = prefix + c.sp_file

    size = os.path.getsize(sp_file)
    count = size // 4
    count = np.minimum(count, buffer_size)
    count = (count // c.batch_size) * c.batch_size

    xs_s = open(x_file, 'rb')
    xs_id_s = open(x_id_file, 'rb')
    xs_seg_s = open(x_seg_file, 'rb')
    ys_mask_s = open(y_mask_file, 'rb')
    ys_id_s = open(y_id_file, 'rb')
    ys_w_s = open(y_w_file, 'rb')
    sps_s = open(sp_file, 'rb')

    b = xs_s.read()
    b_s = c.max_seq_len * count
    xs = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
    xs = xs.reshape((-1, c.max_seq_len))
    xs_s.close()

    b = xs_id_s.read()
    b_s = c.max_seq_len * count
    xs_id = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
    xs_id = xs_id.reshape((-1, c.max_seq_len))
    xs_id_s.close()

    b = xs_seg_s.read()
    b_s = c.max_seq_len * count
    xs_seg = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
    xs_seg = xs_seg.reshape((-1, c.max_seq_len))
    xs_seg_s.close()

    b = ys_mask_s.read()
    b_s = c.max_mask_len * count
    ys_mask = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
    ys_mask = ys_mask.reshape((-1, c.max_mask_len))
    ys_mask_s.close()

    b = ys_id_s.read()
    b_s = c.max_mask_len * count
    ys_id = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
    ys_id = ys_id.reshape((-1, c.max_mask_len))
    ys_id_s.close()

    b = ys_w_s.read()
    b_s = c.max_mask_len * count
    ys_w = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
    ys_w = ys_w.reshape((-1, c.max_mask_len))
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
    tf_dataset = tf_dataset.batch(batch_size=c.batch_size)

    runtime = time.time() - begin

    if c.debug:
        print("BLOCK_{0} with {1} in {2:.3} s.".format(block_id, count, runtime))

    return tf_dataset


def load_fine_tuning_block(target, block_id, buffer_size):
    global c

    begin = time.time()

    prefix = target + "block_{0}/".format(block_id)
    x_file = prefix + c.x_file
    x_id_file = prefix + c.x_id_file
    x_seg_file = prefix + c.x_seg_file
    y_file = prefix + c.y_file

    size = os.path.getsize(x_file)
    count = size // (4 * c.max_seq_len)
    count = np.minimum(count, buffer_size)
    count = (count // c.batch_size) * c.batch_size

    xs_s = open(x_file, 'rb')
    xs_id_s = open(x_id_file, 'rb')
    xs_seg_s = open(x_seg_file, 'rb')
    ys_s = open(y_file, 'rb')

    b = xs_s.read()
    b_s = c.max_seq_len * count
    xs = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
    xs = xs.reshape((-1, c.max_seq_len))
    xs_s.close()

    b = xs_id_s.read()
    b_s = c.max_seq_len * count
    xs_id = np.frombuffer(buffer=b, count=b_s, dtype=np.float32)
    xs_id = xs_id.reshape((-1, c.max_seq_len))
    xs_id_s.close()

    b = xs_seg_s.read()
    b_s = c.max_seq_len * count
    xs_seg = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
    xs_seg = xs_seg.reshape((-1, c.max_seq_len))
    xs_seg_s.close()

    b = ys_s.read()
    # b_s = c.max_mask_len * count
    b_s = 1 * count
    ys = np.frombuffer(buffer=b, count=b_s, dtype=np.int32)
    # ys = ys.reshape((-1, c.max_mask_len))
    ys = ys.reshape((-1, 1))
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
    tf_dataset = tf_dataset.batch(batch_size=c.batch_size)

    runtime = time.time() - begin

    if c.debug:
        print("BLOCK_{0} with {1} in {2:.3} s.".format(block_id, count, runtime))

    return tf_dataset


##

def pre_train_model():
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.train or c.train_buffer_size == 0:
        return

    with strategy.scope():

        @tf.function
        def distributed_train_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
            def per_gpu_train_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
                enc_padding_mask = create_mask(x_id)

                with tf.GradientTape() as tape:
                    x_enc = model(x, enc_padding_mask, x_seg)
                    y_hat_mask, y_hat_sp = model.classify(x_enc, y_id)
                    loss = loss_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)
                    mask_accuracy, label_accuracy = accuracy_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                return loss, mask_accuracy, label_accuracy

            l1, a1, a2 = strategy.run(per_gpu_train_step, args=(x, x_id, x_seg, y_mask, y_id, y_w, sp))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return l1, a1, a2

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count
            acc_a2 = tf.reduce_sum(a2.values, axis=-1) / c.gpu_count

            return acc_l1, acc_a1, acc_a2

        @tf.function
        def distributed_test_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
            def per_gpu_test_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
                enc_padding_mask = create_mask(x_id)

                x_enc = model(x, enc_padding_mask, x_seg)
                y_hat_mask, y_hat_sp = model.classify(x_enc, y_id)
                loss = loss_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)
                mask_accuracy, label_accuracy = accuracy_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)

                return loss, mask_accuracy, label_accuracy

            l1, a1, a2 = strategy.run(per_gpu_test_step, args=(x, x_id, x_seg, y_mask, y_id, y_w, sp))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return l1, a1, a2

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count
            acc_a2 = tf.reduce_sum(a2.values, axis=-1) / c.gpu_count

            return acc_l1, acc_a1, acc_a2

        template = '\n E: {} ({:.2f}%) | Loss: [{:.4f}, {:.4f}] | Mask / Label Acc: [{:.4f}, {:.4f}, {:.4f}, {:.4f}] | delta = {:.2f} \n'

        def load_train_block(b_id):
            train_data = load_pre_training_block(c.train_data_dir, b_id, c.train_buffer_size)
            train_dataset = strategy.experimental_distribute_dataset(train_data)
            return train_dataset

        def load_test_block(b_id):
            test_data = load_pre_training_block(c.test_data_dir, b_id, c.test_buffer_size)
            test_dataset = strategy.experimental_distribute_dataset(test_data)
            return test_dataset

        tr_steps = (c.train_buffer_size * c.train_blocks) // c.batch_size
        va_steps = (c.test_buffer_size * c.test_blocks) // c.batch_size

        tf_train_dataset, tf_test_dataset = None, None

        for e in range(c.epochs):
            start = time.time()
            tr_l1_acc, tr_a1_acc, tr_a2_acc, tr_step = 0, 0, 0, 0

            for block_id in range(c.train_blocks):
                if c.train_blocks > 1:
                    tf_train_dataset = load_train_block(block_id)
                else:
                    if e <= 0:
                        tf_train_dataset = load_train_block(block_id)

                for x, x_id, x_seg, y_mask, y_id, y_w, sp in tf_train_dataset:
                    l1, a1, a2 = distributed_train_step(x, x_id, x_seg, y_mask, y_id, y_w, sp)
                    tr_l1_acc, tr_a1_acc, tr_a2_acc, tr_step = tr_l1_acc + l1, tr_a1_acc + a1, tr_a2_acc + a2, tr_step + 1
                    l1_mu, a1_mu, a2_mu = tr_l1_acc / tr_step, tr_a1_acc / tr_step, tr_a2_acc / tr_step
                    percent = 1e2 * (tr_step / tr_steps)
                    printf("TRAINING : {} {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", block_id, tr_step, percent, l1_mu, a1_mu, a2_mu)

            if e <= 0:
                tf_test_dataset = load_test_block(0)

            va_l1_acc, va_a1_acc, va_a2_acc, va_step = 0, 0, 0, 0

            for x, x_id, x_seg, y_mask, y_id, y_w, sp in tf_test_dataset:
                l1, a1, a2 = distributed_test_step(x, x_id, x_seg, y_mask, y_id, y_w, sp)
                va_l1_acc, va_a1_acc, va_a2_acc, va_step = va_l1_acc + l1, va_a1_acc + a1, va_a2_acc + a2, va_step + 1
                l1_mu, a1_mu, a2_mu = va_l1_acc / va_step, va_a1_acc / va_step, va_a2_acc / va_step
                percent = 1e2 * (va_step / va_steps)
                printf("VALIDATION : {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", va_step, percent, l1_mu, a1_mu, a2_mu)

            tr_l1_acc, tr_a1_acc, tr_a2_acc = tr_l1_acc / tr_step, tr_a1_acc / tr_step, tr_a2_acc / tr_step
            va_l1_acc, va_a1_acc, va_a2_acc = va_l1_acc / va_step, va_a1_acc / va_step, va_a2_acc / va_step

            delta, percent = time.time() - start, (e / c.epochs) * 1e2
            printf(template.format(e, percent, tr_l1_acc, va_l1_acc, tr_a1_acc, tr_a2_acc, va_a1_acc, va_a2_acc, delta))

            if (e + 1) % c.checkpoint_factor == 0:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(e + 1, ckpt_save_path))

    return


def pre_train_inference(dataset, validation_file, buffer_size, blocks):
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.infer or buffer_size == 0:
        return

    with strategy.scope():
        s = open(validation_file, mode='w', encoding='utf-8')
        row_format = "{0}\t{1}\t{2}\t{3}\n"
        batch_format = "{0}\t{1}\t{2}\n"

        def persist(batch, y_hat, y_sp, y_mask, sp, a1, a2):
            if not c.debug:
                return 0, 0, 0, 0

            y_mask_len = np.int32(y_w.numpy().sum(axis=1))

            y_hat_max = np.int32(tf.argmax(y_hat, axis=2).numpy())
            y_sp_max = np.int32(tf.argmax(y_sp, axis=1).numpy())
            y_mask_max = np.int32(y_mask.numpy())
            sp_max = np.int32(sp.numpy())

            mask_accuracy, label_accuracy = a1.numpy(), a2.numpy()

            sp_count, sp_acc, n_sp_count, n_sp_acc = 0, 0, 0, 0

            for q in range(c.batch_size):
                y_mask_len_q = y_mask_len[q]
                y_hat_max_q = list(map(str, y_hat_max[q][:y_mask_len_q]))
                y_mask_max_q = list(map(str, y_mask_max[q][:y_mask_len_q]))
                y_hat_max_q = ' '.join(y_hat_max_q)
                y_mask_max_q = ' '.join(y_mask_max_q)
                y_sp_max_q = y_sp_max[q]
                sp_max_q = sp_max[q]

                row_content = row_format.format(y_hat_max_q, y_mask_max_q, y_sp_max_q, sp_max_q)
                s.write(row_content)

                match_sp, match_n_sp = sp_max_q == y_sp_max_q == 1, sp_max_q == y_sp_max_q == 0
                sp_c, n_sp_c = sp_max_q == 1, sp_max_q == 0

                sp_count, n_sp_count = sp_count + sp_c, n_sp_count + n_sp_c,
                sp_acc, n_sp_acc = sp_acc + match_sp, n_sp_acc + match_n_sp

            batch_content = batch_format.format(batch, mask_accuracy, label_accuracy)
            s.write(batch_content)

            return sp_count, sp_acc, n_sp_count, n_sp_acc

        @tf.function
        def distributed_inference_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
            def inference_step(x, x_id, x_seg, y_mask, y_id, y_w, sp):
                enc_padding_mask = create_mask(x_id)

                x_enc = model(x, enc_padding_mask, x_seg, training=False)
                y_hat_mask, y_hat_sp = model.pre_train_classify(x_enc, y_id)
                loss = loss_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)
                mask_accuracy, label_accuracy = accuracy_function(y_mask, y_w, y_hat_mask, sp, y_hat_sp)

                return y_hat_mask, y_hat_sp, loss, mask_accuracy, label_accuracy

            y_hat, y_sp, l1, a1, a2 = strategy.run(inference_step, args=(x, x_id, x_seg, y_mask, y_id, y_w, sp))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return y_hat, y_sp, l1, a1, a2

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count
            acc_a2 = tf.reduce_sum(a2.values, axis=-1) / c.gpu_count

            return y_hat, y_sp, acc_l1, acc_a1, acc_a2

        l1_acc, a1_acc, a2_acc, steps = 0, 0, 0, (buffer_size * blocks) // c.batch_size
        batch, sp_count, sp_acc, n_sp_count, n_sp_acc = 0, 1e-7, 0, 1e-7, 0

        for b in range(blocks):
            train_dataset = load_pre_training_block(dataset, b, buffer_size)
            train_dataset = strategy.experimental_distribute_dataset(train_dataset)

            for x, x_id, x_seg, y_mask, y_id, y_w, sp in train_dataset:
                y_hat, y_sp, l1, a1, a2 = distributed_inference_step(x, x_id, x_seg, y_mask, y_id, y_w, sp)
                l1_acc, a1_acc, a2_acc, batch = l1_acc + l1, a1_acc + a1, a2_acc + a2, batch + 1
                l1_mu, a1_mu, a2_mu = l1_acc / batch, a1_acc / batch, a2_acc / batch
                percent = 1e2 * (batch / steps)

                sp_c, match_sp, n_sp_c, match_n_sp = persist(batch, y_hat, y_sp, y_mask, sp, a1, a2)
                sp_count, n_sp_count = sp_count + sp_c, n_sp_count + n_sp_c,
                sp_acc, n_sp_acc = sp_acc + match_sp, n_sp_acc + match_n_sp

                printf("INFERENCE : {} ({:.3}%) L1 = {:.4} A1 = {:.4} A2 = {:.4} ", batch, percent, l1_mu, a1_mu, a2_mu)

        l1_acc, a1_acc, a2_acc = l1_acc / batch, a1_acc / batch, a2_acc / batch

        sp_rel, n_sp_rel = sp_acc / sp_count, n_sp_acc / n_sp_count
        footer_format = "\nTEST : L1 = {:.4} : A1 / A2 = {:.4} {:.4} : SP / N_SP = {:.4} ({}) {:.4} ({}) of {:.8} {:.8}"
        footer = footer_format.format(l1_acc, a1_acc, a2_acc, sp_rel, sp_acc, n_sp_rel, n_sp_acc, sp_count, n_sp_count)
        s.write(footer)
        print(footer)
        s.close()

    return


##


def fine_tune_loss_function(y, y_hat):
    global c, token_loss_func, sp_loss_func

    if mode == 'GENERATIVE':
        tokens = tf.one_hot(y, c.vocab_size, dtype=tf.float32)
        token_loss_ps = token_loss_func(tokens, y_hat)
        token_loss_ps = tf.reduce_sum(token_loss_ps)
        return token_loss_ps

    if mode == 'DISCRIMINATIVE':
        dc = tf.one_hot(y, 2, dtype=tf.float32)
        dc_loss_ps = sp_loss_func(dc, y_hat)
        dc_loss_ps = tf.reduce_sum(dc_loss_ps)
        return dc_loss_ps


def fine_tune_accuracy_function(y, y_hat):
    if mode == 'GENERATIVE':
        tokens = tf.cast(tf.argmax(y_hat, 2), dtype=tf.float32)
        y_mask = tf.cast(y, tf.float32)
        y_error = tf.abs(tokens - y_mask)
        y_error = y_error / (y_error + 1e-15)
        y_error = 1 - tf.reduce_mean(y_error)
        return y_error

    if mode == 'DISCRIMINATIVE':
        y_dc = tf.one_hot(y, 2, dtype=tf.float32)
        alpha = tf.reduce_mean(tf.abs(tf.round(y_hat) - y_dc))
        dc_accuracy = 1 - alpha
        return dc_accuracy


def fine_tune_model():
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.train or c.train_buffer_size == 0:
        return

    with strategy.scope():

        @tf.function
        def distributed_train_step(x, x_id, x_seg, y):
            def per_gpu_train_step(x, x_id, x_seg, y):
                enc_padding_mask = create_mask(x_id)

                with tf.GradientTape() as tape:
                    x_enc = model(x, enc_padding_mask, x_seg)
                    y_hat = model.fine_tune_classify(x_enc)
                    loss = fine_tune_loss_function(y=y, y_hat=y_hat)
                    mask_accuracy = fine_tune_accuracy_function(y=y, y_hat=y_hat)

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                return loss, mask_accuracy

            l1, a1 = strategy.run(per_gpu_train_step, args=(x, x_id, x_seg, y))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return l1, a1

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count

            return acc_l1, acc_a1

        @tf.function
        def distributed_test_step(x, x_id, x_seg, y):
            def per_gpu_test_step(x, x_id, x_seg, y):
                enc_padding_mask = create_mask(x_id)

                x_enc = model(x, enc_padding_mask, x_seg)
                y_hat = model.classify(x_enc)
                loss = fine_tune_loss_function(y=y, y_hat=y_hat)
                mask_accuracy = fine_tune_accuracy_function(y=y, y_hat=y_hat)

                return loss, mask_accuracy

            l1, a1 = strategy.run(per_gpu_test_step, args=(x, x_id, x_seg, y))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return l1, a1

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count

            return acc_l1, acc_a1

        template = '\n E: {} ({:.2f}%) | Loss: {:.4f} {:.4f} | Mask [{:.4f}, {:.4f}] | delta = {:.2f} \n'

        def load_train_block(b_id):
            train_data = load_fine_tuning_block(c.train_data_dir, b_id, c.train_buffer_size)
            train_dataset = strategy.experimental_distribute_dataset(train_data)
            return train_dataset

        def load_test_block(b_id):
            test_data = load_fine_tuning_block(c.test_data_dir, b_id, c.test_buffer_size)
            test_dataset = strategy.experimental_distribute_dataset(test_data)
            return test_dataset

        tr_steps = (c.train_buffer_size * c.train_blocks) // c.batch_size
        va_steps = (c.test_buffer_size * c.test_blocks) // c.batch_size

        tf_train_dataset, tf_test_dataset = None, None

        for e in range(c.epochs):
            start = time.time()
            tr_l1_acc, tr_a1_acc, tr_step = 0, 0, 0

            for block_id in range(c.train_blocks):
                if c.train_blocks > 1:
                    tf_train_dataset = load_train_block(block_id)
                else:
                    if e <= 0:
                        tf_train_dataset = load_train_block(block_id)

                for x, x_id, x_seg, y in tf_train_dataset:
                    l1, a1 = distributed_train_step(x, x_id, x_seg, y)
                    tr_l1_acc, tr_a1_acc, tr_step = tr_l1_acc + l1, tr_a1_acc + a1, tr_step + 1
                    l1_mu, a1_mu = tr_l1_acc / tr_step, tr_a1_acc / tr_step
                    percent = 1e2 * (tr_step / tr_steps)
                    printf("TRAINING : {} {} ({:.3}%) L1 = {:.4} A1 = {:.4}", block_id, tr_step, percent, l1_mu, a1_mu)

            if e <= 0:
                tf_test_dataset = load_test_block(0)

            va_l1_acc, va_a1_acc, va_step = 0, 0, 0

            for x, x_id, x_seg, y in tf_test_dataset:
                l1, a1 = distributed_test_step(x, x_id, x_seg, y)
                va_l1_acc, va_a1_acc, va_step = va_l1_acc + l1, va_a1_acc + a1, va_step + 1
                l1_mu, a1_mu = va_l1_acc / va_step, va_a1_acc / va_step
                percent = 1e2 * (va_step / va_steps)
                printf("VALIDATION : {} ({:.3}%) L1 = {:.4} A1 = {:.4}", va_step, percent, l1_mu, a1_mu)

            tr_l1_acc, tr_a1_acc = tr_l1_acc / tr_step, tr_a1_acc / tr_step
            va_l1_acc, va_a1_acc = va_l1_acc / va_step, va_a1_acc / va_step

            delta, percent = time.time() - start, (e / c.epochs) * 1e2
            printf(template.format(e, percent, tr_l1_acc, va_l1_acc, tr_a1_acc, va_a1_acc, delta))

            # if (e + 1) % c.checkpoint_factor == 0:
            #     ckpt_save_path = ckpt_manager.save()
            #     print('Saving checkpoint for epoch {} at {}'.format(e + 1, ckpt_save_path))

    return


def fine_tune_inference(dataset, validation_file, buffer_size, blocks):
    global c, strategy, model, mask_loss, optimizer, ckpt_manager

    if not c.infer or buffer_size == 0:
        return

    with strategy.scope():
        s = open(validation_file, mode='w', encoding='utf-8')
        row_format = "{0}\t{1}\n"
        batch_format = "B={0}\tA={1}\n"

        def persist_generative(batch, y_hat, y, a1):
            if not c.debug:
                return 0, 0, 0, 0

            y_hat_max = np.int32(tf.argmax(y_hat, axis=2).numpy())
            y_max = np.int32(y.numpy())
            mask_accuracy = a1.numpy()

            batch_content = batch_format.format(batch, mask_accuracy)
            s.write(batch_content)

            for q in range(c.batch_size):
                y_hat_max_q = list(map(str, y_hat_max[q]))
                y_max_q = list(map(str, y_max[q]))
                y_hat_max_q = ' '.join(y_hat_max_q)
                y_mask_max_q = ' '.join(y_max_q)

                row_content = row_format.format(y_hat_max_q, y_mask_max_q)
                s.write(row_content)

            return

        def persist_discriminative(batch, y_hat, y, a1):
            if not c.debug:
                return 0, 0, 0, 0

            y_hat_max = tf.squeeze(tf.argmax(y_hat, axis=-1))
            y_hat_max = np.int32(y_hat_max.numpy())
            y_max = np.int32(np.squeeze(y.numpy()))
            mask_accuracy = a1.numpy()

            batch_content = batch_format.format(batch, mask_accuracy)
            s.write(batch_content)

            for q in range(c.batch_size):
                y_hat_max_q = str(y_hat_max[q])
                y_max_q = str(y_max[q])
                row_content = row_format.format(y_hat_max_q, y_max_q)
                s.write(row_content)

            return

        @tf.function
        def distributed_inference_step(x, x_id, x_seg, y):
            def inference_step(x, x_id, x_seg, y):
                enc_padding_mask = create_mask(x_id)

                x_enc = model(x, enc_padding_mask, x_seg)
                y_hat = model.fine_tune_classify(x_enc)
                loss = fine_tune_loss_function(y=y, y_hat=y_hat)
                mask_accuracy = fine_tune_accuracy_function(y=y, y_hat=y_hat)

                return y_hat, loss, mask_accuracy

            y_hat, l1, a1 = strategy.run(inference_step, args=(x, x_id, x_seg, y))

            replicated = strategy.num_replicas_in_sync > 1

            if not replicated:
                return y_hat, l1, a1

            acc_l1 = tf.reduce_sum(l1.values, axis=-1) / c.gpu_count
            acc_a1 = tf.reduce_sum(a1.values, axis=-1) / c.gpu_count

            return y_hat, acc_l1, acc_a1

        l1_acc, a1_acc, steps = 0, 0, (buffer_size * blocks) // c.batch_size
        batch = 0

        for b in range(blocks):
            train_dataset = load_fine_tuning_block(dataset, b, buffer_size)
            train_dataset = strategy.experimental_distribute_dataset(train_dataset)

            for x, x_id, x_seg, y in train_dataset:
                y_hat, l1, a1 = distributed_inference_step(x, x_id, x_seg, y)
                l1_acc, a1_acc, batch = l1_acc + l1, a1_acc + a1, batch + 1
                l1_mu, a1_mu = l1_acc / batch, a1_acc / batch
                percent = 1e2 * (batch / steps)

                if mode == 'GENERATIVE':
                    persist_generative(batch, y_hat, y, a1)
                if mode == 'DISCRIMINATIVE':
                    persist_discriminative(batch, y_hat, y, a1)

                printf("INFERENCE : {} ({:.3}%) L1 = {:.4} A1 = {:.4}", batch, percent, l1_mu, a1_mu)

        l1_acc, a1_acc = l1_acc / batch, a1_acc / batch

        footer_format = "\nTEST : L1 = {:.4} : A1 = {:.4}"
        footer = footer_format.format(l1_acc, a1_acc)
        s.write(footer)
        print(footer)
        s.close()

    return


##


def run_bert():
    global c
    load_configuration()
    build_model()
    if c.pre_training:
        pre_train_model()
        pre_train_inference(c.train_data_dir, c.train_validation_file, c.train_buffer_size, c.train_blocks)
        pre_train_inference(c.test_data_dir, c.test_validation_file, c.test_buffer_size, c.test_blocks)
    else:
        fine_tune_model()
        fine_tune_inference(c.train_data_dir, c.train_validation_file, c.train_buffer_size, c.train_blocks)
        fine_tune_inference(c.test_data_dir, c.test_validation_file, c.test_buffer_size, c.test_blocks)
    return


if __name__ == "__main__":
    begin = time.time()
    run_bert()
    runtime = (time.time() - begin)
    print("BERT NETWORK in {0} s".format(runtime))
