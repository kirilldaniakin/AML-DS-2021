# -*- coding: utf-8 -*-
"""Copy of Copy of text_classification_rnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15iLQiom8VF7Fn8ixocXimm9OM2HmJIj_

# Text classification with an LSTM
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from context import scripts
import scripts
import tensorflow as tf
import datetime

from keras import backend as K

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


# %load_ext tensorboard
if __name__ == '__main__':
  # get data and max name length
  train_dataset, test_dataset, max_len = scripts.get_data(data_path="../data")
  # creating vocabulary
  VOCAB_SIZE = 54 # 26 lower + 26 upper + 2 special
  encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
      max_tokens=VOCAB_SIZE, standardize=None, output_sequence_length=max_len)
  encoder.adapt(train_dataset.map(lambda text, label: text))

  vocab = np.array(encoder.get_vocabulary())
  print(vocab)

  # for encoder example go to hw1.ipynb in notebooks
  #encoded_example = encoder(example).numpy()
  #print(encoded_example[:5])
  #for n in range(3):
  #  print("Original: ", example[n].numpy())
  #  print("Round-trip: ", "".join(vocab[encoded_example[n]]))
  #  print()

  # LSTM model
  model = tf.keras.Sequential([
      encoder,
      tf.keras.layers.Embedding(
          input_dim=len(encoder.get_vocabulary()),
          output_dim=5,
          # Use masking to handle the variable sequence lengths
          #mask_zero=True
          ),
      tf.keras.layers.LSTM(5),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  # For HP tuning see hw1.ipynb notebook in notebooks folder (hw1.ipynb.txt)
 
  # Train the baseline model and save
  logdir = "logs/scalars/" + "baseline"
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
  print("baseline train:")
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy', f1_score])
  history = model.fit(train_dataset, epochs=100,
                      validation_data=test_dataset,
                      validation_steps=30, callbacks=[tensorboard_callback])
  print("baseline summary:")
  model.summary()
  model.save('baseline')

  # save tuned LSTM
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy', f1_score])

    
  # best tuned LSTM (see hw1.ipynb)  
  logdir = "logs/scalars/classic_lr_" + str(1e-4) + "_eps_" + str(5) 
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
  history = model.fit(train_dataset, epochs=5,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback])

  model.save('custom_lstm')

  # Commented out IPython magic to ensure Python compatibility.
  # %tensorboard --logdir logs/scalars

  classic_model = tf.keras.Sequential([
      encoder,
      tf.keras.layers.Embedding(
          input_dim=len(encoder.get_vocabulary()),
          output_dim=5,
          input_length=max_len
          ),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='sigmoid'),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(8, activation='tanh'),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  # For HP tuning see notebook hw1

  # save tuned model (looking at tensorboard, this model is the "best" on test set from HP tuning)
  classic_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy', f1_score])
  logdir = "logs/scalars/classic_lr_" + str(1e-3) + "_eps_" + str(20) 
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_images=True)
  history = classic_model.fit(train_dataset, epochs=20,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback])
  print("calssic model summary:")
  classic_model.summary()
  classic_model.save('classic_model')

  # Commented out IPython magic to ensure Python compatibility.
  # %tensorboard --logdir logs/scalars
