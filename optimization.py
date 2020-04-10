import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    alpha_1 = None
    alpha_2 = None
    alpha_3 = None

    hidden_size = None

    training_steps = None
    warmup_steps = None
    decay_steps = None

    power = None

    def __init__(self, alpha_1, alpha_2, hidden_size, training_steps, warmup_steps, decay_steps, power):
        super(LearningRateScheduler, self).__init__()
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.hidden_size = hidden_size
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.power = power

    def __call__(self, step):
        current_step = step % self.training_steps
        relative_decay = current_step / self.decay_steps
        polynomial_decayed_alpha = (self.alpha_1 - self.alpha_2) * ((1 - relative_decay) ** self.power) + self.alpha_2

        is_warmup = tf.cond(tf.less(current_step, self.warmup_steps), lambda: 1.0, lambda: 0.0)
        relative_warmup = tf.minimum(current_step / self.warmup_steps, 1.0)
        warmup_alpha = self.alpha_1 * relative_warmup
        self.alpha_3 = (1.0 - is_warmup) * polynomial_decayed_alpha + is_warmup * warmup_alpha

        return self.alpha_3


class CustomOptimizer(optimizer_v2.OptimizerV2):

    def __init__(self, learning_rate, beta_1, beta_2, epsilon, decay, name='Optimizer', **kwargs):
        super(CustomOptimizer, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', decay)
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(CustomOptimizer, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        decay = self._get_hyper('decay', var_dtype)

        lr = apply_state[(var_device, var_dtype)]['lr_t']

        apply_state[(var_device, var_dtype)].update(dict(
                lr=lr,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
                decay=decay
        ))

    def _resource_apply_dense(self, grad, var, apply_state):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        beta_1_power = coefficients['beta_1_power']
        beta_2_power = coefficients['beta_2_power']
        lr_t = coefficients['lr_t']
        beta_1_t = coefficients['beta_1_t']
        beta_2_t = coefficients['beta_2_t']
        epsilon = coefficients['epsilon']
        decay = coefficients['decay']

        m.assign(beta_1_t * m + (1.0 - beta_1_t) * grad)
        v.assign(beta_2_t * v + (1.0 - beta_2_t) * grad * grad)

        var.assign_sub(lr_t * (m / (tf.sqrt(v) + epsilon) + decay * var))

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        m = self.get_slot(handle, 'm')
        v = self.get_slot(handle, 'v')

        pass

    def get_config(self):
        config = super(CustomOptimizer, self).get_config()
        config.update({
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'decay'        : self._serialize_hyperparameter('decay'),
                'beta_1'       : self._serialize_hyperparameter('beta_1'),
                'beta_2'       : self._serialize_hyperparameter('beta_2'),
                'epsilon'      : self.epsilon
        })
        return config
