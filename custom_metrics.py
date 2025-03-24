import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# 1. Dice Coefficient Metric
def dice_coefficient(y_true, y_pred):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice = (2. * intersection + 1.) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + 1.)
    return dice

# 2. Callback for Dice Coefficient
class DiceCallback(Callback):
    def __init__(self, threshold=0.7):
        super(DiceCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_dice = logs.get('dice_coefficient')
        if current_dice is None:
            warnings.warn("Dice coefficient is missing in logs", RuntimeWarning)
        if current_dice > self.threshold:
            self.model.stop_training = True
            print(f"\nReached {self.threshold:.2f} Dice coefficient, stopping training.")

# 3. Learning Rate Scheduler
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=3):
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

lr_scheduler = step_decay_schedule(initial_lr=1e-4, decay_factor=0.2, step_size=3)

# 4. Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# 5. Dice Coefficient per Channel
def dice_coefficient_per_channel(y_true, y_pred, epsilon=1e-6):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0,1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(0,1,2))
    dsc = (2. * intersection + epsilon) / (union + epsilon)
    return dsc

# 6. Channel Specific Dice Coefficient Metrics
def dice_coefficient_channel_0(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[0]

def dice_coefficient_channel_1(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[1]

def dice_coefficient_channel_2(y_true, y_pred):
    return dice_coefficient_per_channel(y_true, y_pred)[2]