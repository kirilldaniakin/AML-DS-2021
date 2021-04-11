import numpy as np

import tensorflow as tf
import datetime
# %load_ext tensorboard
if __name__ == '__main__':
  baseline = tf.keras.models.load_model("baseline.tf")
  custom = tf.keras.models.load_model("custom_lstm.tf")
  classic = tf.keras.models.load_model("classic.tf")
  
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
