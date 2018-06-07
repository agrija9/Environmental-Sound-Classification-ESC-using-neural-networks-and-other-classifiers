# coding= UTF-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

# Load data
X = np.load("feat.npy")
y = np.load('label.npy').ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 233)

# Neural Network Construction
model = Sequential()

# Network Architecture
model.add(Conv1D(64, 3, activation='relu', input_shape = (193, 1)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Most used loss function for muliple-class classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Convert label to onehot encoding
y_train = keras.utils.to_categorical(y_train - 1, num_classes=10) # Converts a class vector (integers) to binary class matrix
y_test = keras.utils.to_categorical(y_test - 1, num_classes=10)

# Make 2-dim into 3-dim array for input for model training
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Train Network
model.fit(X_train, y_train, batch_size=64, epochs=100) # Epochs are tunable

# Compute Accuracy and Loss
score, acc = model.evaluate(X_test, y_test, batch_size=16)

print('Test score:', score)
print('Test accuracy:', acc)