import json
import tensorflow as tf


class Configuration(object):
    # checkpoints
    checkpoint_factor = None
    checkpoint_folder = None
    vocab_folder = None

    # data set
    train_data_dir = None
    test_data_dir = None

    # pre-training
    x_file = None
    x_id_file = None
    x_seg_file = None
    y_mask_file = None
    y_id_file = None
    y_w_file = None
    sp_file = None

    # fine-tuning
    y_file = None

    # validation
    train_validation_file = None
    test_validation_file = None
    cpu_threads = None

    # model
    max_seq_len = None
    mask_prob = None
    num_layers = None
    hidden_size = None
    intermediate_size = None
    num_heads = None
    drop_rate = None

    # training
    batch_size = None
    train_buffer_size = None
    test_buffer_size = None
    blocks = None
    epochs = None
    learning_rate = None

    # mode
    pre_training = None
    reload = None
    freeze = None
    freeze_interval = None

    # execution
    gpu_mode = None
    train = None
    infer = None
    eager = None
    debug = None
    strategy = None

    # gpu
    physical_gpu_count = None
    logical_gpu_count = None
    b_p_gpu = None

    def __init__(self, configuration_file):
        with open(configuration_file) as stream:
            self.configuration = json.load(stream)

        # checkpoints
        self.checkpoint_factor = int(self.configuration["data"]["checkpoint_factor"])
        print("CHECKPOINT_FACTOR = {}".format(self.checkpoint_factor))
        self.checkpoint_folder = self.configuration["data"]["checkpoint_folder"]
        print("CHECKPOINT_FOLDER = {}".format(self.checkpoint_folder))
        self.vocab_folder = self.configuration["data"]["vocab_folder"]
        print("VOCAB_FOLDER = {}".format(self.vocab_folder))

        # data set
        self.train_data_dir = self.configuration["data"]["train_data_dir"]
        print("TRAN_DATA_DIR = {}".format(self.train_data_dir))
        self.test_data_dir = self.configuration["data"]["test_data_dir"]
        print("TEST_DATA_DIR = {}".format(self.test_data_dir))

        # pre-training
        self.x_file = "x.set"
        print("X_FILE = {}".format(self.x_file))
        self.x_id_file = "x_id.set"
        print("X_ID_FILE = {}".format(self.x_id_file))
        self.x_seg_file = "x_seg.set"
        print("X_SEG_FILE = {}".format(self.x_seg_file))
        self.y_mask_file = "y_mask.set"
        print("Y_MASK_FILE = {}".format(self.y_mask_file))
        self.y_id_file = "y_id.set"
        print("Y_ID_FILE = {}".format(self.y_id_file))
        self.y_w_file = "y_w.set"
        print("Y_W_FILE = {}".format(self.y_w_file))
        self.sp_file = "sp.set"
        print("SP_FILE = {}".format(self.sp_file))

        # fine-tuning
        self.y_file = "y.set"
        print("Y_FILE = {}".format(self.y_file))

        # validation
        self.train_validation_file = self.configuration["data"]["train_validation_file"]
        print("TRAIN_VALIDATION_FILE = {}".format(self.train_validation_file))
        self.test_validation_file = self.configuration["data"]["test_validation_file"]
        print("TEST_VALIDATION_FILE = {}".format(self.test_validation_file))
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
        self.train_buffer_size = self.configuration["training"]["train_buffer_size"]
        print("TRAIN_BUFFER_SIZE = {}".format(self.train_buffer_size))
        self.test_buffer_size = self.configuration["training"]["test_buffer_size"]
        print("TEST_BUFFER_SIZE = {}".format(self.test_buffer_size))
        self.train_blocks = self.configuration["training"]["train_blocks"]
        print("TRAIN_BLOCKS = {}".format(self.train_blocks))
        self.test_blocks = self.configuration["training"]["test_blocks"]
        print("TEST_BLOCKS = {}".format(self.test_blocks))
        self.epochs = self.configuration["training"]["epochs"]
        print("EPOCHS = {}".format(self.epochs))
        self.learning_rate = self.configuration["training"]["learning_rate"]
        print("LEARNING_RATE = {}".format(self.learning_rate))

        # mode
        self.pre_training = self.configuration["mode"]["pre_training"]
        print("PRE_TRAINING = {}".format(self.pre_training))
        self.reload = self.configuration["mode"]["reload"]
        print("RELOAD = {}".format(self.reload))
        self.freeze = self.configuration["mode"]["freeze"]
        print("FREEZE = {}".format(self.freeze))
        self.freeze_interval = int(self.configuration["mode"]["freeze_interval"])
        print("FREEZE_INTERVAL = {}".format(self.freeze_interval))

        # execution
        self.gpu_mode = self.configuration["execution"]["gpu_mode"]
        print("GPU_MODE = {}".format(self.gpu_mode))
        self.train = self.configuration["execution"]["train"]
        print("TRAIN = {}".format(self.train))
        self.infer = self.configuration["execution"]["infer"]
        print("INFER = {}".format(self.infer))
        self.eager = self.configuration["execution"]["eager"]
        print("EAGER = {}".format(self.eager))
        self.debug = self.configuration["execution"]["debug"]
        print("DEBUG = {}".format(self.debug))

        self.initialize_strategy()
        return

    def initialize_strategy(self):
        tpu, gpu = False, True

        def initialize_tpu():
            tpu_name = "tpu-demo"
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All devices: ", tf.config.list_logical_devices('TPU'))
            ##
            self.strategy = tf.distribute.TPUStrategy(resolver)
            ##
            self.b_p_gpu = int(self.batch_size / 1.0)
            print("BATCHES_TPU = {0}".format(self.b_p_gpu))

        def initialize_gpu():
            try:
                physical_gpus = tf.config.list_physical_devices('GPU')
                self.physical_gpu_count = len(physical_gpus)
                if self.gpu_mode.lstrip("-+").isdigit():
                    gpu_id = int(self.gpu_mode)
                    tf.config.set_visible_devices(physical_gpus[gpu_id], 'GPU')
                logical_gpus = tf.config.list_logical_devices('GPU')
                self.logical_gpu_count = len(logical_gpus)
                print("PHYSICAL_GPUS_COUNT = {0}".format(self.physical_gpu_count))
                print("LOGICAL_GPUS_COUNT = {0}".format(self.logical_gpu_count))
                ##
                self.strategy = tf.distribute.MirroredStrategy()
                ##
                if self.eager:
                    tf.config.experimental_run_functions_eagerly(True)
                    tf.executing_eagerly()
                self.b_p_gpu = int(self.batch_size / self.logical_gpu_count)
                print("BATCHES_PRO_GPU = {0}".format(self.b_p_gpu))
            except RuntimeError as e:
                print(e)
                exit()

        if tpu:
            initialize_tpu()

        if gpu:
            initialize_gpu()

        return
