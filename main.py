import json
import os
import re
import time

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from loss_functions import MaskLoss, SamePaperLoss
from tokenizer import Tokenizer
from transformer import Transformer

checkpoint_folder = None
vocab_folder = None
max_len = None
batch_size = None
buffer_size = None
mask_prob = None
max_pred_per_seq = None
num_layers = None
d_model = None
num_heads = None
dff = None
rate = None
epochs = None
learning_rate = None
data_file = None

train = None
infer = None
eager = None

strategy = None
model = None
optimizer = None
mask_loss = None
ckpt_manager = None


def download_and_tokenize_data():
    data_set = "imdb_reviews"
    data_folder = "data\\imdb"
    vocab_folder = "vocabs\\imdb_small"
    vocab_size = 2 ** 13
    examples, metadata = tfds.load(name=data_set, data_dir=data_folder, with_info=True, as_supervised=True)
    train_examples, test_examples = examples['train'], examples['test']
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((review.numpy() for (review, i) in train_examples),
                                                                        target_vocab_size=vocab_size)
    tokenizer.save_to_file(vocab_folder)
    return


def clean_and_filter_data():
    data_set = "imdb_reviews"
    data_folder = "data\\imdb"

    examples, metadata = tfds.load(name=data_set, data_dir=data_folder, with_info=True, as_supervised=True)
    train_examples, test_examples = examples['train'], examples['test']

    def format_review(review):
        output = review.decode('utf-8')
        output = re.sub(r'<br /><br />|\'', '', output)
        output = re.sub('\\?+', ' ? ', output)
        output = re.sub('\\.+', ' . ', output)
        output = re.sub('!+', ' ! ', output)
        output = re.sub(',+', ' , ', output)
        output = re.sub(':+', ' : ', output)
        output = re.sub('\\"+', ' " ', output)
        output = re.sub(' +', ' ', output)
        output = re.sub('/+', ' / ', output)
        output = re.sub('-+', ' - ', output)
        output = re.sub('\\(+', ' ( ', output)
        output = re.sub('\\)+', ' ) ', output)
        output = re.sub('\\*+', ' * ', output)
        output = re.sub('\\[+', ' [ ', output)
        output = re.sub('\\]+', ' ] ', output)
        output = re.sub('#+', ' # ', output)
        output = re.sub('\\$+', ' $ ', output)
        output = re.sub('@+', ' @ ', output)
        output = re.sub(' +', ' ', output)
        return output

    def filter(review):
        # non_asci = len(re.findall('[^\x00-\x7F]+', review))
        return True

    data = [(sample.numpy(), index.numpy()) for (sample, index) in train_examples]
    reviews = [format_review(review) for (review, index) in data]
    reviews = [review for review in reviews if filter(review)]

    review_size = 50
    reviews = reviews[0:review_size]

    tokenizer = Tokenizer(vocab_folder)
    reviews = [" ".join(tokenizer.tokenize_seq(review)) for review in reviews]

    with open('data/imdb/reviews.txt', mode='wt', encoding='utf-8') as stream:
        stream.write('\n'.join(reviews))

    def split_review(review):
        output = re.findall('[a-z][\\w\\s,\\"\\-#]+[.|?|!]+', review)
        return output

    with open('data/imdb/reviews.txt', encoding='utf-8') as f:
        content = f.readlines()
    reviews = [x.strip() for x in content]

    transposed_sentences = [split_review(review) for review in reviews]
    transposed_sentences = [sentences for sentences in transposed_sentences if len(sentences) > 0]

    review_meta = [(review_index, len(sentences)) for (review_index, sentences) in enumerate(transposed_sentences)]

    with open('data/imdb/reviews_meta.txt', mode='w', encoding='utf-8') as stream:
        for (review_index, len_sentences) in review_meta:
            stream.write(str(review_index))
            stream.write(';')
            stream.write(str(len_sentences))
            stream.write('\n')
        stream.close()

    sentences = [(review_index, sentence_index, sentence) for (review_index, sentences) in
                 enumerate(transposed_sentences) for (sentence_index, sentence) in enumerate(sentences)]

    with open('data/imdb/sentences.txt', mode='w', encoding='utf-8') as stream:
        for (review_index, sentence_index, sentence) in sentences:
            stream.write(str(review_index))
            stream.write(';')
            stream.write(str(sentence_index))
            stream.write(';')
            stream.write(sentence)
            stream.write('\n')
        stream.close()

    sentences = []
    with open('data/imdb/sentences.txt', 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = int(row[0]), int(row[1]), row[2]
            sentences.append(row)

    reviews_meta = []
    with open('data/imdb/reviews_meta.txt', 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = int(row[0]), int(row[1])
            reviews_meta.append(row)

    return


def load_and_prepare_data():
    sentences_file = "data/imdb/sentences.txt"
    pair_sentences_file = 'data/imdb/pair_sentences.txt'
    repeat_coefficient = 1

    reviews_meta = []
    with open('data/imdb/reviews_meta.txt', 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = int(row[0]), int(row[1])
            reviews_meta.append(row)
    reviews_len = len(reviews_meta)

    sentences = []
    with open(sentences_file, 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = int(row[0]), int(row[1]), row[2]
            sentences.append(row)

    map_sentences = {}
    for review_index, sentence_index, sentence in sentences:
        map_sentences[(review_index, sentence_index)] = sentence

    dataset = []
    for r_x, s_x, sen_x in sentences:
        r_meta_x = reviews_meta[r_x]
        r_len_x = r_meta_x[1]
        for y in range(repeat_coefficient):
            combine = np.random.uniform(0.0, 1.0, 1)
            delta = 0.5
            r_y, = np.random.randint(0, reviews_len, 1, dtype=np.int32)
            if combine > delta and s_x < r_len_x - 1:
                sen_y = map_sentences[(r_x, s_x + 1)]
                sen_x = sen_x.strip()
                sen_y = sen_y.strip()
                sample = (sen_x, sen_y, 1)
            else:
                r_meta_y = reviews_meta[r_y]
                r_len_y = r_meta_y[1]
                s_y, = np.random.randint(0, r_len_y, 1)
                sen_y = map_sentences[(r_y, s_y)]
                sen_x = sen_x.strip()
                sen_y = sen_y.strip()
                sample = (sen_x, sen_y, 0)
            dataset.append(sample)

    with open(pair_sentences_file, mode='w', encoding='utf-8') as stream:
        for (sen_x, sen_y, label) in dataset:
            stream.write(sen_x)
            stream.write(';')
            stream.write(sen_y)
            stream.write(';')
            stream.write(str(label))
            stream.write('\n')
        stream.close()

    print("PAIRS = {0}".format(len(dataset)))

    pair_sentences = []
    with open(pair_sentences_file, 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = row[0], row[1], int(row[2])
            pair_sentences.append(row)

    return


def encode_and_quantize():
    global max_len, mask_prob, max_pred_per_seq

    pair_sentences_file = 'data/imdb/pair_sentences.txt'
    appended_sequences_file = 'data/imdb/appended_sequences.txt'
    masked_sequences_file = 'data/imdb/masked_sequences.txt'
    tokenized_sequences_file = 'data/imdb/tokenized_sequences.txt'

    tokenizer = Tokenizer(vocab_folder)

    INITIAL_SYM = "[CLS]"
    INITIAL_TOKEN = tokenizer.encode_word(INITIAL_SYM)
    SEP_SYM = "[SEP]"
    SEP_TOKEN = tokenizer.encode_word(SEP_SYM)
    MASK_SYM = "[MASK]"
    MASK_TOKEN = tokenizer.encode_word(MASK_SYM)

    pair_sentences = []
    with open(pair_sentences_file, 'r', encoding='utf-8') as stream:
        for sentence in stream:
            row = sentence.split(";")
            row = row[0], row[1], int(row[2])
            pair_sentences.append(row)

    raw_sequence_dataset = []
    masked_dataset = []
    tokenized_dataset = []
    count = 0
    for sen_x, sen_y, label in pair_sentences:
        raw_sequence = INITIAL_SYM + " " + sen_x + " " + SEP_SYM + " " + sen_y + " " + SEP_SYM
        raw_sequence_dataset.append(raw_sequence)

        sen_x_split = sen_x.split(" ")
        sen_y_split = sen_y.split(" ")
        raw_sequence = [INITIAL_SYM] + sen_x_split + [SEP_SYM] + sen_y_split + [SEP_SYM]
        sequence_len_run = len(raw_sequence)

        masked_len = min(int(mask_prob * sequence_len_run), max_pred_per_seq)
        indices = [index for (index, word) in enumerate(raw_sequence) if word not in [INITIAL_SYM, SEP_SYM]]
        np.random.shuffle(indices)
        masked_indices = indices[0:masked_len]
        masked_indices.sort()
        masked_sequence = raw_sequence.copy()
        masked_words = []
        for masked_index in masked_indices:
            masking, = np.random.uniform(0.0, 1.0, 1)
            if masking >= 0.8:
                masked_words.append(masked_sequence[masked_index])
                masked_sequence[masked_index] = MASK_SYM
            else:
                masked_words.append(masked_sequence[masked_index])
                masked_sequence[masked_index] = MASK_SYM
        raw_train_sample = (masked_sequence, masked_words, masked_indices)
        masked_dataset.append(raw_train_sample)

        tokenized_sen = [tokenizer.encode_word(word) for word in masked_sequence]
        tokenized_seq_len = len(tokenized_sen)
        seq_y_s = tokenized_sen.index(SEP_TOKEN)
        tokenized_seq_x_len = seq_y_s + 1
        tokenized_seq_y_len = tokenized_seq_len - tokenized_seq_x_len

        if tokenized_seq_len > max_len:
            continue

        if len(masked_sequence) != len(tokenized_sen):
            print("ERROR SEQ {0} != {1}".format(len(masked_sequence), len(tokenized_sen)))
            continue

        seq_padding = max_len - tokenized_seq_len
        tokenized_sen += [0] * seq_padding
        input_mask = [1] * tokenized_seq_len + [0] * seq_padding
        segment_ids = [0] * tokenized_seq_x_len + [1] * tokenized_seq_y_len + [0] * seq_padding

        mask_padding = max_pred_per_seq - masked_len
        tokenized_mask = [tokenizer.encode_word(word) for word in masked_words]
        mask_weights = [1.0] * masked_len
        if len(masked_indices) != len(tokenized_mask):
            print("ERROR MASK {0} != {1}".format(len(masked_indices), len(tokenized_mask)))
            continue

        tokenized_mask += [0] * mask_padding
        masked_indices += [0] * mask_padding
        mask_weights += [0.0] * mask_padding

        tokenized_train_sample = (tokenized_sen, input_mask, segment_ids, masked_indices, tokenized_mask, mask_weights, label)
        tokenized_dataset.append(tokenized_train_sample)
        print("{0}".format(count))
        count += 1

    dataset_len = len(tokenized_dataset)
    sequences_len = len(pair_sentences)
    print("SEQUENCES / PAIRS = {0} / {1}".format(dataset_len, sequences_len))

    with open(appended_sequences_file, mode='w', encoding='utf-8') as stream:
        for index in range(dataset_len):
            appended = raw_sequence_dataset[index]
            stream.write(appended)
            stream.write('\n')
        stream.close()

    with open(masked_sequences_file, mode='w', encoding='utf-8') as stream:
        for index in range(dataset_len):
            masked_sequence, masked_words, masked_indices = masked_dataset[index]
            for word in masked_sequence:
                stream.write(word)
                stream.write(" ")
            stream.write(";")
            for word in masked_words:
                stream.write(word)
                stream.write(" ")
            stream.write(";")
            for masked_index in masked_indices:
                stream.write(str(masked_index))
                stream.write(" ")
            stream.write('\n')
        stream.close()

    with open(tokenized_sequences_file, mode='w', encoding='utf-8') as stream:
        for index in range(dataset_len):
            tokenized_sen, input_mask, segment_ids, masked_indices, tokenized_mask, mask_weights, label = tokenized_dataset[index]
            for token in tokenized_sen:
                stream.write(str(token))
                stream.write(" ")
            stream.write(";")
            for token in input_mask:
                stream.write(str(token))
                stream.write(" ")
            stream.write(";")
            for token in segment_ids:
                stream.write(str(token))
                stream.write(" ")
            stream.write(";")
            for mask_index in masked_indices:
                stream.write(str(mask_index))
                stream.write(" ")
            stream.write(";")
            for token in tokenized_mask:
                stream.write(str(token))
                stream.write(" ")
            stream.write(";")
            for weight in mask_weights:
                stream.write(str(weight))
                stream.write(" ")
            stream.write(";")
            stream.write(str(label))
            stream.write('\n')
        stream.close()

    return


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_mask(input_mask):
    # Encoder padding mask
    dim_0 = input_mask.shape[0]
    dim_1 = input_mask.shape[1]
    broadcast = tf.ones((dim_0, dim_1, 1), dtype=np.float32)
    mask = tf.reshape(input_mask, shape=(dim_0, 1, dim_1))
    # enc_padding_mask = create_padding_mask(inp)
    enc_padding_mask = broadcast * mask
    return enc_padding_mask


def loss_function(y_mask, y_mask_w, y_hat_mask, label, ns_output):
    global batch_size, max_pred_per_seq

    tokens = tf.argmax(y_hat_mask, 2)
    same_paper = tf.argmax(ns_output, 1)

    y_mask_w = tf.cast(y_mask_w, tf.int64)
    y_mask = tf.cast(y_mask, tf.int64)
    label = tf.cast(label, tf.int64)
    mask_accuracy = tf.abs(tokens * y_mask_w - y_mask)
    mask_accuracy = 1 - tf.where(mask_accuracy > 0).shape[0] / tf.reduce_sum(y_mask_w)
    label_accuracy = 1 - tf.reduce_sum(tf.abs(same_paper - label)) / batch_size

    return train_loss, mask_accuracy, label_accuracy


def load_configuration():
    global checkpoint_folder, vocab_folder, max_len, batch_size, buffer_size, \
        mask_prob, max_pred_per_seq, num_layers, d_model, num_heads, dff, rate, epochs, learning_rate, data_file, train, infer, eager

    with open("configuration.json") as fp:
        configuration = json.load(fp)

    checkpoint_folder = configuration["checkpoint_folder"]
    print("CHECKPOINT_FOLDER = {0}".format(checkpoint_folder))
    vocab_folder = configuration["vocab_folder"]
    print("VOCAB_FOLDER = {0}".format(vocab_folder))
    max_len = configuration["max_len"]
    print("MAX_LEN = {0}".format(max_len))
    batch_size = configuration["batch_size"]
    print("BATCH_SIZE = {0}".format(batch_size))
    buffer_size = configuration["buffer_size"]
    print("BUFFER_SIZE = {0}".format(buffer_size))
    mask_prob = configuration["mask_prob"]
    print("MASK_PROB = {0}".format(mask_prob))
    max_pred_per_seq = int(mask_prob * max_len)
    print("MAX_PRED_PER_SEQ = {0}".format(max_pred_per_seq))
    num_layers = configuration["num_layers"]
    print("NUM_LAYERS = {0}".format(num_layers))
    d_model = configuration["d_model"]
    print("D_MODEL = {0}".format(d_model))
    num_heads = configuration["num_heads"]
    print("NUM_HEADS = {0}".format(num_heads))
    dff = configuration["dff"]
    print("DFF = {0}".format(dff))
    rate = configuration["rate"]
    print("RATE = {0}".format(rate))
    epochs = configuration["epochs"]
    print("EPOCHS = {0}".format(epochs))
    learning_rate = configuration["learning_rate"]
    print("LEARNING_RATE = {0}".format(learning_rate))
    data_file = configuration["data_file"]
    print("DATA_FILE = {0}".format(data_file))
    train = configuration["train"]
    print("TRAIN = {0}".format(train))
    infer = configuration["infer"]
    print("INFERENCE = {0}".format(infer))
    eager = configuration["eager"]
    print("EAGER = {0}".format(eager))

    if eager:
        tf.config.experimental_run_functions_eagerly(True)
        tf.executing_eagerly()


def build_model():
    global max_len, strategy, model, optimizer, ckpt_manager, mask_loss

    strategy = tf.distribute.MirroredStrategy()

    tokenizer = Tokenizer(vocab_folder)
    vocab_size = tokenizer.vocab_size

    with strategy.scope():
        model = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            input_vocab_size=vocab_size,
                            target_vocab_size=vocab_size,
                            rate=rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-6)

        mask_loss = MaskLoss(batch_size)
        same_paper_loss = SamePaperLoss(batch_size)

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_folder, max_to_keep=5)

        @tf.function
        def mm(y_true, y_pred):
            y_mask, y_mask_w = tf.cast(y_true[:, 0, :], dtype=tf.int32), y_true[:, 1, :]
            y_hat_mask = y_pred
            tokens = tf.argmax(y_hat_mask, 2)
            y_mask_w = tf.cast(y_mask_w, tf.int64)
            y_mask = tf.cast(y_mask, tf.int64)
            mask_accuracy = tf.abs(tokens * y_mask_w - y_mask)
            mask_accuracy = 1 - tf.reduce_sum(tf.cast(mask_accuracy > 0, dtype=tf.int64)) / tf.reduce_sum(y_mask_w)
            return mask_accuracy

        @tf.function
        def spm(y_true, y_pred):
            label = y_true
            y_hat_ns = y_pred
            same_paper = tf.reshape(tf.argmax(y_hat_ns, 1), shape=(batch_size, 1))
            label = tf.cast(label, tf.int64)
            label_accuracy = 1 - tf.reduce_sum(tf.abs(same_paper - label)) / batch_size
            return label_accuracy

        model.compile(loss={"output_1": mask_loss, "output_2": same_paper_loss},
                      optimizer=optimizer,
                      metrics={"output_1": mm, "output_2": spm})

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return


def load_data():
    global data_file, batch_size, buffer_size, max_len, max_pred_per_seq

    tokenized_sequences = np.ones((buffer_size, max_len), dtype=np.float32)
    input_masks = np.ones((buffer_size, max_len), dtype=np.float32)
    segments_ids = np.ones((buffer_size, max_len), dtype=np.int32)
    masks_indices = np.ones((buffer_size, max_pred_per_seq), dtype=np.int32)
    masks_weights = np.ones((buffer_size, max_pred_per_seq), dtype=np.float32)
    tokenized_masks = np.ones((buffer_size, max_pred_per_seq), dtype=np.int32)
    labels = np.ones((buffer_size, 1), dtype=np.int32)

    with open(data_file, 'r', encoding='utf-8') as stream:
        sequences = stream.readlines()
        stream.close()

    def assert_sequence_length(width, index):
        if width != max_len:
            print("ERROR at {} got {} != is {}".format(index, width, max_len))
            exit(1)

    def assert_mask_length(width, index):
        if width != max_pred_per_seq:
            print("ERROR at {} got {} != is {}".format(index, width, max_pred_per_seq))
            exit(1)

    for index, sentence in enumerate(sequences[0:buffer_size]):
        tokenized_sequence, input_mask, segment_ids, mask_indices, tokenized_mask, mask_weights, label = sentence.split(";")

        tokenized_sequence = list(map(float, tokenized_sequence.split(" ")[:-1]))
        assert_sequence_length(len(tokenized_sequence), index)
        tokenized_sequences[index] = tokenized_sequence

        input_mask = list(map(float, input_mask.split(" ")[:-1]))
        assert_sequence_length(len(input_mask), index)
        input_masks[index] = input_mask

        segment_ids = list(map(int, segment_ids.split(" ")[:-1]))
        assert_sequence_length(len(segment_ids), index)
        segments_ids[index] = segment_ids

        mask_indices = list(map(int, mask_indices.split(" ")[:-1]))
        assert_mask_length(len(mask_indices), index)
        masks_indices[index] = mask_indices

        mask_weights = list(map(float, mask_weights.split(" ")[:-1]))
        assert_mask_length(len(mask_weights), index)
        masks_weights[index] = mask_weights

        tokenized_mask = list(map(int, tokenized_mask.split(" ")[:-1]))
        assert_mask_length(len(tokenized_mask), index)
        tokenized_masks[index] = tokenized_mask

        label = int(label)
        labels[index] = label

    return tokenized_sequences, input_masks, segments_ids, masks_indices, tokenized_masks, masks_weights, labels


def train_model():
    global strategy, model, optimizer, ckpt_manager, data_file, train, max_len, batch_size, buffer_size, epochs

    if not train:
        return

    tokenized_sequences, input_masks, segments_ids, masks_indices, tokenized_masks, masks_weights, labels = load_data()
    model.fit(
            x=[tokenized_sequences, input_masks, segments_ids, masks_indices],
            y=[np.stack((tokenized_masks, masks_weights), axis=1), labels], epochs=epochs, batch_size=batch_size)


def inference():
    global model, optimizer, train_loss, train_accuracy, ckpt_manager, data_file, infer, buffer_size

    if not infer:
        return

    tokenized_sequences = []
    input_masks = []
    segments_ids = []
    masks_indices = []
    masks_weights = []
    tokenized_masks = []
    labels = []

    with open(data_file, 'r', encoding='utf-8') as stream:
        sequences = stream.readlines()
        stream.close()

    for sentence in sequences:
        if len(tokenized_sequences) >= buffer_size:
            break

        tokenized_sequence, input_mask, segment_ids, mask_indices, tokenized_mask, mask_weights, label = sentence.split(";")

        tokenized_sequence = list(map(float, tokenized_sequence.split(" ")[:-1]))
        tokenized_sequences.append(tokenized_sequence)

        input_mask = list(map(float, input_mask.split(" ")[:-1]))
        input_masks.append(input_mask)

        segment_ids = list(map(int, segment_ids.split(" ")[:-1]))
        segments_ids.append(segment_ids)

        mask_indices = list(map(int, mask_indices.split(" ")[:-1]))
        masks_indices.append(mask_indices)

        mask_weights = list(map(float, mask_weights.split(" ")[:-1]))
        masks_weights.append(mask_weights)

        tokenized_mask = list(map(int, tokenized_mask.split(" ")[:-1]))
        tokenized_masks.append(tokenized_mask)

        label = int(label)
        labels.append(label)

    tf_tokenized_sequences = tf.data.Dataset.from_tensor_slices(tokenized_sequences)
    tf_tokenized_sequences = tf_tokenized_sequences.batch(batch_size)
    tf_input_masks = tf.data.Dataset.from_tensor_slices(input_masks)
    tf_input_masks = tf_input_masks.batch(batch_size)
    tf_segments_ids = tf.data.Dataset.from_tensor_slices(segments_ids)
    tf_segments_ids = tf_segments_ids.batch(batch_size)
    tf_mask_indices = tf.data.Dataset.from_tensor_slices(masks_indices)
    tf_mask_indices = tf_mask_indices.batch(batch_size)
    tf_tokenized_masks = tf.data.Dataset.from_tensor_slices(tokenized_masks)
    tf_tokenized_masks = tf_tokenized_masks.batch(batch_size)
    tf_masks_weights = tf.data.Dataset.from_tensor_slices(masks_weights)
    tf_masks_weights = tf_masks_weights.batch(batch_size)
    tf_labels = tf.data.Dataset.from_tensor_slices(labels)
    tf_labels = tf_labels.batch(batch_size)

    tf_train_dataset = tf.data.Dataset.zip(
            (tf_tokenized_sequences, tf_input_masks, tf_segments_ids, tf_mask_indices, tf_tokenized_masks, tf_masks_weights, tf_labels))

    for tokenized_sequence, input_mask, segment_ids, mask_indices, tokenized_mask, mask_weights, label in tf_train_dataset:
        arr = tokenized_sequence.numpy()

    validation = []

    for batch, (tokenized_sequence, input_mask, segment_ids, mask_indices, tokenized_mask, mask_weights, label) in enumerate(
            tf_train_dataset):
        enc_padding_mask = create_mask(input_mask)
        y_hat_mask, y_hat_ns = model(tokenized_sequence, enc_padding_mask, input_mask, segment_ids, mask_indices)
        err, mask_accuracy, label_accuracy = loss_function(tokenized_mask, mask_weights, y_hat_mask, label, y_hat_ns)
        tokens = tf.argmax(y_hat_mask, axis=2)
        same_paper = tf.argmax(y_hat_ns, axis=1)
        mask_len = mask_weights.numpy().sum(axis=1)
        entry = (
                tokenized_mask.numpy(), tokens.numpy(), mask_len, label.numpy(), same_paper.numpy(), mask_accuracy.numpy(),
                label_accuracy.numpy())
        validation.append(entry)

    validation.sort(key=lambda x: x[5] + x[6], reverse=True)

    validation_file = "data/formulas/validation.txt"

    with open(validation_file, mode='w', encoding='utf-8') as stream:
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
    load_configuration()
    # download_and_tokenize_data()
    # clean_and_filter_data()
    # load_and_prepare_data()
    # encode_and_quantize()
    build_model()
    train_model()
    # inference()
    return


if __name__ == "__main__":
    load_configuration()
    a = time.time()
    # run_tokenizer()
    run_bert()
    b = (time.time() - a) * 1e3
    print("BERT NETWORK in {0}".format(b))
