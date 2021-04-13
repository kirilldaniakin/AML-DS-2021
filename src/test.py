import numpy as np

import tensorflow as tf
import datetime
# %load_ext tensorboard
from context import scripts
import scripts

import matplotlib.pyplot as plt

from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_score(y_true, y_pred):
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# %load_ext tensorboard
if __name__ == '__main__':
  # get data and max name length
  train_dataset, test_dataset, max_len = scripts.get_data(data_path="../data")
  baseline = tf.keras.models.load_model("baseline", custom_objects={'f1_score':f1_score})
  baseline.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-03),
              metrics=['accuracy', f1_score])
  custom = tf.keras.models.load_model("custom_lstm", custom_objects={'f1_score':f1_score})
  classic.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-03),
              metrics=['accuracy', f1_score])
  classic = tf.keras.models.load_model("classic_model", custom_objects={'f1_score':f1_score})
  classic.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-03),
              metrics=['accuracy', f1_score])
  
  test_loss, test_acc, f1 = baseline.evaluate(test_dataset)

  print('baseline Test Loss:', test_loss)
  print('baseline Test Accuracy:', test_acc)
  print('baseline Test F1:', f1)
  
  test_loss, test_acc, f1 = custom.evaluate(test_dataset)

  print('custom lstm Test Loss:', test_loss)
  print('custom lstm Test Accuracy:', test_acc)
  print('custom lstm Test F1:', f1)
  
  test_loss, test_acc, f1 = classic.evaluate(test_dataset)

  print('classic Test Loss:', test_loss)
  print('classic Test Accuracy:', test_acc)
  print('classic Test F1:', f1)
