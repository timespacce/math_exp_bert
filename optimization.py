import tensorflow as tf


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    alpha_1 = None
    alpha_2 = None
    alpha_3 = None

    hidden_size = None
    warmup_steps = None
    decay_steps = None

    power = None

    def __init__(self, alpha_1, alpha_2, hidden_size, warmup_steps, decay_steps, power):
        super(LearningRateScheduler, self).__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.hidden_size = hidden_size
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.power = power

    def __call__(self, step):
        relative_decay = step / self.decay_steps
        polynomial_decayed_alpha = (self.alpha_1 - self.alpha_2) * ((1 - relative_decay) ** self.power) + self.alpha_2

        is_warmup = 1.0 if step <= self.warmup_steps else 0.0
        relative_warmup = step / self.warmup_steps
        warmup_alpha = self.alpha_1 * relative_warmup
        self.alpha_3 = (1.0 - is_warmup) * polynomial_decayed_alpha + is_warmup * warmup_alpha

        return self.alpha_3
