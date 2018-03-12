import os
import csv
from keras import backend as K
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import math
import numpy as np
from PIL import Image         
import cv2                 
import matplotlib.pyplot as plt
from os import getcwd
import csv
import keras

import tensorflow as tf
tf.python.control_flow_ops = tf

samples = []

# collect examples from the collected data
with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_nobarrier_turn/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_right_turn_next_to_water/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_corrective_before_bridge/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_smoothly_curve_right_water/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data0/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_smooth_corners_anti_clockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_one_lap_center_clockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_second_lap_center_clockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_smoothly_around_curves/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_full_run_model_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_redo_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_one_lap_counterClockwise/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# augmented special case data 

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_unique_bpundaries/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_tree_shadow/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('/Users/hdayanidhi/CarND-Behavioral-Cloning-P3/data_unique_dirt_boundary_and_tree_shadow/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=16):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                image_flipped = np.fliplr(center_image)
                measurement_flipped = -center_angle
                left_image = cv2.imread(batch_sample[1])
                left_measurement = center_angle + 0.2
                right_image = cv2.imread(batch_sample[2])
                right_measurement = center_angle - 0.2 
                images.append(center_image)
                angles.append(center_angle)
                images.append(image_flipped)
                angles.append(measurement_flipped)
                images.append(left_image)
                angles.append(left_measurement)
                images.append(right_image)
                angles.append(right_measurement)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            yield (X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#set up the model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 127.5) - 1.0))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.50))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# model training
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=10)
model.save('model.h5')
