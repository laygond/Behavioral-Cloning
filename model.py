from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import csv
import random
from math import ceil

def shadow(img):
    """
    Add random shadow to your image by using y=mx+b principal.
    Add a slope line inside the image, then dim one side of
    the image while keeping the other side intact. 
    """
    img = np.copy(img)
    h, w = img.shape[0:2]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    m = h / (x2 - x1)
    b = - m * x1    # find intercept 0=mx+b
    for i in range(h):
        x = int((i - b) / m)    # from i=mx+b
        img[i, :x, :] = (img[i, :x, :] * .5).astype(np.uint8)

    return img


def generator(samples, batch_size=32):
    """
    This is a python generator function (yields rather than return).
    Provides features and labels in small groups (a group at a time).
    Every group is augmented in data 12x batch_size through preprocessing. 
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                angle_correction = [0.0, 0.25, -0.25] # center, left, right
                for camera_position in range(3):    # center, left, right
                    # Load feature and label
                    img_path = '../MYDATA/IMG/'+batch_sample[camera_position].split('/')[-1]
                    image = cv2.imread(img_path)
                    angle = float(batch_sample[3]) + angle_correction[camera_position]
                    
                    # Add and Augment Data
#                     # Normal
#                     images.append(image)
#                     angles.append(angle)
#                     # Flipped
#                     images.append(cv2.flip(image,1))
#                     angles.append(angle*-1.0)
                    # Shadow
                    images.append(shadow(image))
                    angles.append(angle)
                    #Flipped with Shadow
                    images.append(shadow(cv2.flip(image,1)))
                    angles.append(angle*-1.0)


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


            
# READ CSV File
samples = []
with open('../MYDATA/custom.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Load DATASET: Training and Validation         
random.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)            

# Define generator functions
batch_size=32
train_generator      = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# #Explore generator structure 
# gen_list = []
# gen_list = next(train_generator)
# print (gen_list[1][0])       # First steering angle value
# print (gen_list[0][0].shape) # Shape of first image
# img = cv2.cvtColor(gen_list[0][0],cv2.COLOR_BGR2RGB)
# plt.figure()#figsize=(11, 5))   #width, height in inches
# plt.imshow(img)
# plt.show()

# Load or Create model
if("sexy_model3.h5" in os.listdir(".")):
    # Load Model
    model = load_model("sexy_model3.h5")
else:
    # Model Net
    model = Sequential()
    ch, row, col = 3, 160, 320
    model.add(Cropping2D(cropping=((50, 20), (0,0)), input_shape =(row,col,ch)))
    model.add(Lambda(lambda x: x/127.5 - 1.)) # Normalized centered around zero with small standard deviation
    model.add(Convolution2D (24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D (36,5,5,subsample=(2,2),activation= "relu"))
    model.add(Convolution2D (48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D (64,3,3,activation="relu"))
    model.add(Convolution2D (64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(lr = 1e-4), metrics=['acc'])

# Train Model Net
history_object = model.fit_generator(train_generator, 
                    steps_per_epoch=ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=ceil(len(validation_samples)/batch_size), 
                    epochs=60, verbose=1)

# Save model
model.save('sexy_model3.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'], label="train_loss")
plt.plot(history_object.history['val_loss'], label="val_loss")
plt.plot(history_object.history['acc'], label="train_acc")
plt.plot(history_object.history['val_acc'], label="val_acc")
plt.title('Training MSE Loss and Accuracy')
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch #')
plt.legend()
plt.savefig("README_images/graph", bbox_inches='tight')
#plt.show()


