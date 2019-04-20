# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers

### DATA PREP ###

# Load data
(x_raw_train, y_raw_train), (x_raw_test, y_raw_test) = imdb.load_data(num_words=10000)

# Word index
word_index = imdb.get_word_index()
num_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode one review
' '.join([num_index.get(x - 3, '?') for x in x_raw_train[0]])

# Function to vectorize squences
def vectorize_squences(sequences, size):
	mat = np.zeros((len(sequences), size))
	for i, seq in enumerate(sequences):
		mat[i, seq] = 1 # Change all values for one sequence at once
	return mat

# Vectorize
x_train = vectorize_squences(x_raw_train, size=10000)
x_test = vectorize_squences(x_raw_test, size=10000)

# Convert y to float
y_train = y_raw_train.astype('float32')
y_test = y_raw_test.astype('float32')

### MODEL ###

# Create model
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='rmsprop', 
			  loss='binary_crossentropy', 
			  metrics=['accuracy']) 

# Fit model
history = model.fit(x_train, y_train, 
					epochs=20, 
					batch_size=512, 
					validation_data=(x_test, y_test))

# Function to review traning and validation loss
def review_history(history):
    history_dict = history.history
    train_loss_values = history_dict['loss']
    test_loss_values = history_dict['val_loss']
    epochs = range(1, len(train_loss_values) + 1)
    plt.plot(epochs, train_loss_values, 'bo', label='Training loss')
    plt.plot(epochs, test_loss_values, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Review history
review_history(history)







