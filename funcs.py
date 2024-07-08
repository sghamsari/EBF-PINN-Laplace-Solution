# --------------- Define custom layer ---------------
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
# --------------- Custom Layer ---------------
class real_variable(tf.keras.layers.Layer):
    def __init__(self, nvariable=1, trainable=True, name=None, initial_value=None, multiplier=1, seed=None, complex=False, kernel_min_max=None):
        super(real_variable, self).__init__(name=name)
        self.nvariable = nvariable
        self.trainable = trainable
        self.initial_value = initial_value
        self.multiplier = multiplier
        self.seed = seed
        self.complex = complex
        self.kernel_min_max = kernel_min_max
    def build(self, input_shape):
        if self.initial_value == None:
            if isinstance(self.kernel_min_max, list):
                if len(self.kernel_min_max) > 2:
                    print(f' Warning kernel_min_max should have 2 elements but it has {len(self.kernel_min_max)} elements!')
                    kernel_min_value = self.kernel_min_max[0]
                    kernel_max_value = self.kernel_min_max[1]
                elif len(self.kernel_min_max) < 2:
                    raise ValueError('Number of kernel_min_max elements should at least 2')
                else:
                    kernel_min_value, kernel_max_value = self.kernel_min_max
            else:
                kernel_min_value, kernel_max_value = [-0.05, 0.05]
            tf.random.set_seed(self.seed)
            self.w = self.add_weight(
                shape=(self.nvariable,),
                initializer=tf.keras.initializers.RandomUniform(seed=self.seed, minval=kernel_min_value, maxval=kernel_max_value),
                trainable=self.trainable,
                dtype=tf.float64
            )
            if self.complex == 'complex':
                self.w2 = self.add_weight(
                    shape=(self.nvariable,),
                    initializer=tf.keras.initializers.RandomUniform(seed=self.seed, minval=kernel_min_value, maxval=kernel_max_value),
                    trainable=self.trainable,
                    dtype=tf.float64
                )
        else:
            try:
                self.w = self.add_weight(
                    shape=(self.nvariable,),
                    initializer=self.initial_value,
                    trainable=self.trainable,
                    dtype=tf.float64
                )
                if self.complex == 'complex':
                    self.w2 = self.add_weight(
                        shape=(self.nvariable,),
                        initializer=self.initial_value,
                        trainable=self.trainable,
                        dtype=tf.float64
                    )
            except:
                self.w = self.add_weight(
                    shape=(self.nvariable,),
                    initializer='ones',
                    trainable=self.trainable,
                    dtype=tf.float64
                )
                if self.complex == 'complex':
                    self.w2 = self.add_weight(
                        shape=(self.nvariable,),
                        initializer='ones',
                        trainable=self.trainable,
                        dtype=tf.float64
                    )
                self.w.assign(self.w * self.initial_value)
                if self.complex == 'complex':
                    self.w2.assign(self.w2 * self.initial_value)
        if self.multiplier == 'auto':
            tf.random.set_seed(None)
            self.multiplier = self.add_weight(
                shape=(1,),
                initializer='random_normal',
                trainable=self.trainable,
                dtype=tf.float64
            )
            self.w.assign(self.w * self.multiplier)
        elif isinstance(self.multiplier, (int, float)):
            self.w.assign(self.w * self.multiplier)
        else:
            raise ValueError('Unknown multiplier input')
    def call(self, inputs):
        # if self.complex == True:
        #     return tf.cast(self.w, dtype=tf.complex128)
        if self.complex in ('img', 'imaginary', 'imag'):
            return tf.complex(np.array([0.0]), self.w)
        elif self.complex in ('real', 'rl'):
            return tf.dtypes.complex(self.w, np.array([0.0]))
        elif self.complex == 'complex':
            return tf.dtypes.complex(self.w, self.w2)
        else:
            return self.w
    def get_config(self):
        config = super(real_variable, self).get_config()
        # config.update({'nVariables':self.nvariable, 'weights':self.w})
        config.update({'nVariables': self.nvariable})
        return config

class complex_variable(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(complex_variable, self).__init__(name=name)
    def build(self, input_shape):
        pass
    def call(self, inputs):
        complex_out = tf.dtypes.complex(inputs[0], inputs[1])
        return complex_out
    def get_config(self):
        config = super(complex_variable, self).get_config()
        config.update({'real_variable':self.inputs[0], 'img_variables':self.inputs[1]})
        return config

# ----- Learning Rate Changer Callback -----
def lr_scheduler(Epochs, initial_lr, final_lr, mode='exp'):
    if any(not isinstance(x, float) for x in (initial_lr, final_lr)):
        raise ValueError('The learning rate should be float number')
    if mode == ('exp' or 'exponential'):  # Exponential
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: final_lr * (initial_lr / final_lr) ** ((Epochs - epoch - 1) / (Epochs - 1)))
    elif mode == ('lin' or 'linear'):  # Linear
        return tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: initial_lr - (initial_lr - final_lr) * epoch / (Epochs - 1))
    else:
        raise ValueError('Unknown lr_scheduler mode. Choose between \'exp\' of \'lin\'')

# ----- Break-Training Callback -----
class StopTraining(tf.keras.callbacks.Callback):
    def __init__(self, loss_check):
        super().__init__()
        self.loss_check = loss_check

    def on_epoch_end(self, epoch, logs={}):
        if tf.abs(logs.get('loss')) < self.loss_check:
            print(f'Training stopped because the loss was lesser than: {self.loss_check}')
            self.model.stop_training = True

# ----- Figure Function -----
def loss_fig(mhist, plt_save=None):
    # --------------- Draw Model Loss - Epochs --------------
    plt.semilogy(mhist.history['loss'], label='loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if not plt_save == None:
        plt.savefig(f'{plt_save}_Loss-Epochs')
    plt.show()
    # --------------- Draw Model Loss - Lr --------------
    plt.semilogy(mhist.history['lr'], mhist.history['loss'], label='loss')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    if not plt_save == None:
        plt.savefig(f'{plt_save}_Loss-Lr')
    plt.show()

# ----- Function to Print the Results ----
class PrintFreq(tf.keras.callbacks.Callback):
    def __init__(self, layer_names=None, print_freq=100):
        super().__init__()
        self.print_freq = print_freq
        self.start_time = 0
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = tf.timestamp()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times += [(tf.timestamp() - self.start_time)]
        epoch_time = (tf.timestamp() - self.start_time).numpy()
        logs['epoch_time'] = epoch_time
        if (epoch + 1) % self.print_freq == 0:  # check if the epoch number is divisible by print_freq
            lr = self.model.optimizer.learning_rate.numpy()
            loss = logs.get('loss')  # get the loss from the logs dictionary
            acc = [f'{i}: {logs.get(i)}' for i in logs.keys() if 'mse' in i]
            Total_epochs = self.params['epochs']
            print(f'Epoch {epoch + 1} / {Total_epochs}, Loss: {loss}, Accuracy: {acc}, lr: {lr}, elapced time:{(tf.timestamp() - self.start_time) * 1000:0.0f} ms')
