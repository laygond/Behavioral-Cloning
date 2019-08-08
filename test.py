from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import csv
import random
from math import ceil

# READ CSV File
samples = []
with open('../MYDATA01/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Load DATASET: Training and Validation         
random.shuffle(samples)

# Load Model
model = load_model('model.h5')

# Testing
for i in range(10):
    center_img_path = '../MYDATA01/IMG/'+samples[i][0].split('/')[-1]
    center_image = cv2.imread(center_img_path)
    center_angle = float(samples[i][3])
    predicted_angle = float(model.predict(center_image[None, :, :, :], batch_size=1))
    print("Original: ", center_angle, " Predicted: ", predicted_angle) 
















