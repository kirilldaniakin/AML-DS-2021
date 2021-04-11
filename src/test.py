import numpy as np

import tensorflow as tf
import datetime
# %load_ext tensorboard
from context import scripts
import scripts

import matplotlib.pyplot as plt


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
  baseline = tf.keras.models.load_model("baseline")
  custom = tf.keras.models.load_model("custom_lstm")
  classic = tf.keras.models.load_model("classic")
  
  test_loss, test_acc = baseline.evaluate(test_dataset)

  print('baseline Test Loss:', test_loss)
  print('baseline Test Accuracy:', test_acc)
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plot_graphs(history, 'accuracy')
  plt.ylim(None, 1)
  plt.subplot(1, 2, 2)
  plot_graphs(history, 'loss')
  plt.ylim(0, None)
  
  test_loss, test_acc = classic.evaluate(test_dataset)

  print('custom lstm Test Loss:', test_loss)
  print('custom lstm Test Accuracy:', test_acc)
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plot_graphs(history, 'accuracy')
  plt.ylim(None, 1)
  plt.subplot(1, 2, 2)
  plot_graphs(history, 'loss')
  plt.ylim(0, None)
  
  test_loss, test_acc = classic.evaluate(test_dataset)

  print('classic Test Loss:', test_loss)
  print('classic Test Accuracy:', test_acc)
  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plot_graphs(history, 'accuracy')
  plt.ylim(None, 1)
  plt.subplot(1, 2, 2)
  plot_graphs(history, 'loss')
  plt.ylim(0, None)
