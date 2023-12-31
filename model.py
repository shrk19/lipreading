from keras.models import load_model
import tensorflow as tf
import os 

def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def get_model():
    model = load_model('model60.h5', compile=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=CTCLoss)
    return model


