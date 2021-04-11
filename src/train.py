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
# %load_ext tensorboard
if __name__ == '__main__':
  # get data and max name length
  train_data, test_data, max_len = scripts.get_data(data_path="../data/SeoulBikeData.csv")

  # creating vocabulary
  VOCAB_SIZE = 54 # 26 lower + 26 upper + 2 special
  encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
      max_tokens=VOCAB_SIZE, standardize=None, output_sequence_length=max_len)
  encoder.adapt(train_dataset.map(lambda text, label: text))

  vocab = np.array(encoder.get_vocabulary())
  print(vocab)

  # for encoder example go to data_prep.ipynb in notebooks
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

  # HP tuning
  # tensorboard_callback will save running logs, so it's possible to see with command: tensorboard --logdir=logs                                                 
  lrs = [1e-3, 1e-4, 1e-5]
  epochs = [5, 10, 20]

  for lr in lrs:  
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(lr),
                metrics=['accuracy'])
    for eps in epochs:
      logdir = "logs/scalars/lr_" + str(lr) + "_eps_" + str(eps) 
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
      history = model.fit(train_dataset, epochs=eps,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback],
                          verbose=0)

  # Train the baseline model and save
  logdir = "logs/scalars/" + "baseline"
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
  print("baseline train:")
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy'])
  history = model.fit(train_dataset, epochs=100,
                      validation_data=test_dataset,
                      validation_steps=30, callbacks=[tensorboard_callback])
  print("baseline summary:")
  model.summary()
  model.save('baseline.tf')

  # save tuned LSTM
  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

  logdir = "logs/scalars/classic_lr_" + str(lr) + "_eps_" + str(1e-4) 
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
  history = classic_model.fit(train_dataset, epochs=5,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback])

  model.save('custom_lstm.tf')

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

  # HP tuning
  lrs = [1e-3, 1e-4, 1e-5]
  epochs = [5, 10, 20]

  for lr in lrs:
    classic_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(lr),
                metrics=['accuracy'])
    for eps in epochs:
      logdir = "logs/scalars/classic_lr_" + str(lr) + "_eps_" + str(eps) 
      tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
      history = classic_model.fit(train_dataset, epochs=eps,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback],
                          verbose=0)

  print("calssic model summary:")
  classic_model.summary()

  # save tuned model (looking at tensorboard, this model is the "best" on test set)
  classic_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(1e-3),
                metrics=['accuracy'])
  logdir = "logs/scalars/classic_lr_" + str(lr) + "_eps_" + str(1e-3) 
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
  history = classic_model.fit(train_dataset, epochs=20,
                          validation_data=test_dataset,
                          validation_steps=30, callbacks=[tensorboard_callback])
  classic_model.save('classic_model.tf')

  # Commented out IPython magic to ensure Python compatibility.
  # %tensorboard --logdir logs/scalars
