# COSC420, Helena Crawford, 13 April 2020
# Neural network for Simple Classification
# Based off examples by Lech Szymanski, Lecture 3
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import gzip
from cosc420_assig1_data import load_data
data_type = 'clean'
(train_images, train_labels), (test_images, test_labels), class_names = load_data(task='simple', dtype=data_type)

# Train with data augmentation on (True) or off (False)
# Expand the dataset with realistic transformations of existing images
optn_data_augmentation = False
# Train with (True) or without (False) dropout in the penultimate layer
# Randomly drop nodes to reduce overfitting by disrupting the architecture
optn_dropout = True
# Use weight decay regularisation (True) or not (False) in the fully connected layers
# Penalize large weights (another sign of overfitting)
optn_weight_decay = False
# Use batch normalisation (True) or not (False) after max pooling layers
# Improve training performance
optn_batch_norm = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

model_name = 'saved/task1_'
model_name += data_type

if optn_data_augmentation:
   model_name += "_aug"

if optn_dropout:
   model_name += "_drop"

if optn_weight_decay:
   model_name += "_wd"

if optn_batch_norm:
   model_name += "_bnorm"

model_save_name = model_name + '.h5'
history_save_name = model_name + '.hist'
model_plot_name = model_name + '.png'

# ************************************************
# * Creating and training a neural network model *
# ************************************************
n_classes = len(class_names)
history = None

if os.path.isfile(model_save_name):
   # ***************************************************
   # * Loading previously trained neural network model *
   # ***************************************************

   # Create a basic model instance
   print("Loading model from %s..." % model_save_name)
   model = tf.keras.models.load_model(model_save_name)

   # Check if history file exists and if so load training history
   if os.path.isfile(history_save_name):
      with gzip.open(history_save_name) as f:
         history = pickle.load(f)

else:

   # Create feed-forward network
   model = tf.keras.models.Sequential()

   # Add convolutional layer, 3x3 window, 2N = 56 filters - specify the size of the input as NxNx3
   # (implicit arguments - step size is [1,1], padding="SAME")
   model.add(tf.keras.layers.Conv2D(56, (3, 3), activation='relu', input_shape=(28, 28, 3), kernel_initializer = 'he_uniform'))

   # Add max pooling layer, 2x2 window
   # (implicit arguments - step size is [2,2], padding="SAME")
   model.add(tf.keras.layers.MaxPooling2D((2, 2)))
   if optn_batch_norm:
      model.add(tf.keras.layers.BatchNormalization())

   # Add convolutional layer, 3x3 window, 4N filters
   # (implicit arguments - step size is [1,1], padding="SAME")
   model.add(tf.keras.layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer='he_uniform'))

   # Add max pooling layer, 2x2 window
   # (implicit arguments - step size is [2,2], padding="SAME")
   model.add(tf.keras.layers.MaxPooling2D((2, 2)))
   if optn_batch_norm:
      model.add(tf.keras.layers.BatchNormalization())

   # Add convolutional layer, 3x3 window, 8N filters
   # (implicit arguments - step size is [1,1], padding="SAME")
   model.add(tf.keras.layers.Conv2D(224, (3, 3), activation='relu', kernel_initializer='he_uniform'))


   # Add max pooling layer, 2x2 window
   # (implicit arguments - step size is [2,2], padding="SAME")
   model.add(tf.keras.layers.MaxPooling2D((2, 2)))
   if optn_batch_norm:
      model.add(tf.keras.layers.BatchNormalization())

   # Flatten the output maps for fully connected layer
   model.add(tf.keras.layers.Flatten())

   if optn_weight_decay:
      regulariser_choice = tf.keras.regularizers.l2(0.01)
   else:
      regulariser_choice = None

   # Add a fully connected layer of 4N neurons
   model.add(tf.keras.layers.Dense(112, activation='relu',
                                   kernel_initializer='he_uniform', bias_initializer='zeros',
                                   kernel_regularizer=regulariser_choice))

   # Add a fully connected layer of 16N neurons
   model.add(tf.keras.layers.Dense(448, activation='relu',
                                   kernel_initializer='he_uniform', bias_initializer='zeros',
                                   kernel_regularizer=regulariser_choice))

   if optn_dropout:
      model.add(tf.keras.layers.Dropout(0.5))

   # Add a fully connected layer with number of output neurons the same
   # as the number of classes
   model.add(tf.keras.layers.Dense(n_classes))

   # Create a diagram of the model and save it to a png file
   tf.keras.utils.plot_model(model, model_plot_name, show_shapes=True)

   # Use cross entropy for training
   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

   # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
   # training
   model.compile(optimizer='adam',
                 loss=loss_fn,
                 metrics=['accuracy'])

   if optn_data_augmentation:
      # Data augmentation
      datagen = tf.keras.preprocessing.image.ImageDataGenerator(
         # epsilon for ZCA whitening - highlight letter features better
         zca_epsilon=1e-06,
         # randomly shift images horizontally (fraction of total width)
         width_shift_range=0.1,
         # randomly shift images vertically (fraction of total height)
         height_shift_range=0.1,
         # set mode for filling points outside the input boundaries
         fill_mode='nearest',
         # rotation range limited to preserve M-W distinction
         rotation_range=45
         )

      datagen.fit(train_images)
      augmented_train_images_and_labels = datagen.flow(train_images, train_labels, batch_size=100)

      # Train the model for 20 epochs
      train_info = model.fit(augmented_train_images_and_labels, validation_data=(test_images, test_labels),
                             epochs=20, shuffle=True)
   else:
      # Train the model for 10 epochs
      train_info = model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                             epochs=10, shuffle=True)

   # Save model to file
   print("Saved model to %s..." % model_save_name)
   model.save(model_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)

# *********************************************************
# * Training history *
# *********************************************************

if history is not None:
   # Plot training and validation accuracy over the course of training
   plt.figure()
   plt.plot(history['accuracy'], label='accuracy')
   plt.plot(history['val_accuracy'], label = 'val_accuracy')
   plt.xlabel('Epoch')
   plt.ylabel('Accuracy')
   plt.ylim([0, 1])
   plt.legend(loc='lower right')
   plt.title(model_save_name)

# *********************************************************
# * Evaluating the neural network model within tensorflow *
# *********************************************************

loss_train, accuracy_train = model.evaluate(train_images,  train_labels, verbose=0)
loss_test, accuracy_test = model.evaluate(test_images, test_labels, verbose=0)

print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)

plt.show()
