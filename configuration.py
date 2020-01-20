import json
import tensorflow as tf


class Configuration(object):
    # data
    checkpoint_folder = None
    vocab_folder = None
    data_file = None
    cpu_threads = None

    # model
    max_len = None
    mask_prob = None
    num_layers = None
    hidden_size = None
    intermediate_size = None
    num_heads = None
    drop_rate = None

    # training
    batch_size = None
    buffer_size = None
    epochs = None
    learning_rate = None

    # execution
    train = None
    infer = None
    eager = None

    # gpu
    gpu_count = None
    b_p_gpu = None

    def __init__(self, configuration_file):
        with open(configuration_file) as stream:
            self.configuration = json.load(stream)

        # data
        self.checkpoint_folder = self.configuration["data"]["checkpoint_folder"]
        print("CHECKPOINT_FOLDER = {}".format(self.checkpoint_folder))
        self.vocab_folder = self.configuration["data"]["vocab_folder"]
        print("VOCAB_FOLDER = {}".format(self.vocab_folder))
        self.data_file = self.configuration["data"]["data_file"]
        print("DATA_FILE = {}".format(self.data_file))
        self.validation_file = self.configuration["data"]["validation_file"]
        print("VALIDATION_FILE = {}".format(self.validation_file))
        self.cpu_threads = self.configuration["data"]["cpu_threads"]
        print("CPU_THREADS = {}".format(self.cpu_threads))

        # model
        self.vocab_size = self.configuration["model"]["vocab_size"]
        print("VOCAB_SIZE = {}".format(self.vocab_size))
        self.max_seq_len = self.configuration["model"]["max_seq_len"]
        print("MAX_SEQUENCE_LEN = {}".format(self.max_seq_len))
        self.mask_prob = self.configuration["model"]["mask_prob"]
        print("MASK_PROB = {}".format(self.mask_prob))
        self.max_mask_len = int(self.max_seq_len * self.mask_prob)
        print("MAX_MASK_LENGTH = {}".format(self.max_mask_len))
        self.num_layers = self.configuration["model"]["num_layers"]
        print("NUM_LAYERS = {}".format(self.num_layers))
        self.hidden_size = self.configuration["model"]["hidden_size"]
        print("HIDDEN_SIZE = {}".format(self.hidden_size))
        self.intermediate_size = self.configuration["model"]["intermediate_size"]
        print("INTERMEDIATE_SIZE = {}".format(self.intermediate_size))
        self.num_heads = self.configuration["model"]["num_heads"]
        print("NUM_HEADS = {}".format(self.num_heads))
        self.drop_rate = self.configuration["model"]["drop_rate"]
        print("DROP_RATE = {}".format(self.drop_rate))

        # training
        self.batch_size = self.configuration["training"]["batch_size"]
        print("BATCH_SIZE = {}".format(self.batch_size))
        self.buffer_size = self.configuration["training"]["buffer_size"]
        print("BUFFER_SIZE = {}".format(self.buffer_size))
        self.epochs = self.configuration["training"]["epochs"]
        print("EPOCHS = {}".format(self.epochs))
        self.learning_rate = self.configuration["training"]["learning_rate"]
        print("LEARNING_RATE = {}".format(self.learning_rate))

        # execution
        self.train = self.configuration["execution"]["train"]
        print("TRAIN = {}".format(self.train))
        self.infer = self.configuration["execution"]["infer"]
        print("INFER = {}".format(self.infer))
        self.eager = self.configuration["execution"]["eager"]
        print("EAGER = {}".format(self.eager))

        # gpu
        if self.eager:
            tf.config.experimental_run_functions_eagerly(True)
            tf.executing_eagerly()
        self.gpu_count = len(tf.config.experimental.list_logical_devices('GPU'))
        print("GPU_COUNT = {0}".format(self.gpu_count))
        self.b_p_gpu = int(self.batch_size / self.gpu_count)
        print("BATCHES_PRO_GPU = {0}".format(self.b_p_gpu))
