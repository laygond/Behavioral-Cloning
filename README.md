# Behavioral Cloning Project

In this project a regression deep neural network is used to clone driving behavior.  The model architecture written in Keras is [Nvidia's CNN](https://arxiv.org/pdf/1604.07316v1.pdf) with dropout to prevent overfitting during training. Image data from three front cameras is used as features and the vehicles steering angles as labels to train the neural network. Once trained, an image fed into the model will output a steering angle to an autonomous vehicle. To collect the data, [Udacity's simulator](https://github.com/udacity/self-driving-car-sim) is used to steer a car around a track. This repo uses [Udacity's CarND-Behavioral-Cloning repo](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for guidance.

[//]: # (List of Images used in this README.md)
[image1]: ./README_images/visualization.gif "Visualization"
[image2]: ./README_images/traffic_sign_catalog.png "Catalog"
[image3]: ./README_images/train_set_dist.png "Training Set Distribution"
[image4]: ./README_images/architecture.png "Model Architecture"
[image5]: ./README_images/NNparam.png "Model Parameters"
[image6]: ./README_images/traffic_signs.png "Traffic Signs"
[image7]: ./README_images/stopinspect.png "Stop Sign Inspect"

![alt text][image1]

## Directory Structure
```
├── CarND-Behavioral-Cloning-P3
│   ├── drive.p                  # test script: driving the car in autonomous mode (makes use of model.h5)
│   ├── model.h5                 # trained Keras model ready for testing or keep training   
│   ├── model.py                 # contains the script to create and train the model (generates model.h5)
│   ├── README_images            # Images used by README.md
|   │   └── ...
│   ├── README.md
|   ├── .gitignore               # git file to prevent unnecessary files from being uploaded
│   ├── test.py
│   └── video.py                 # A script that makes a video from images.
└── MYDATA                       # Data used for training and validation (not included in the repo)
    ├── custom.csv
    ├── driving_log.csv
    ├── IMG
    └── max.csv
```

## The Project's Workflow
The workflow of this project is the following:
* Use the simulator in "training mode" to collect data of good driving behavior and place it in `MYDATA`
* `model.py` uses the data in `MYDATA` to train and validate the model and saves it as `model.h5`
* `drive.py` interacts with the simulator in "autonomous mode" to receive image data
* `drive.py` uses `model.h5` to predict a steering angle from that image data and sends the angle back to the simulator.
Therefore the simulator acts as a client and `drive.py` as the server.

## Udacity's Simulator
The simulator can be downloaded [here](https://github.com/udacity/self-driving-car-sim) and a docker file can be found [here](https://github.com/udacity/CarND-Term1-Starter-Kit). The simulator allows you to chose training mode for collecting data or autonomous mode for testing your model. In either mode there are two tracks you can drive on. For this project we will only focus on track 1. The control display is also shown below.

![alt text][image1] ![alt text][image1]

Additional Information:
- You can takeover in autonomous mode while W or S are held down so you can control the car the same way you would in training mode. This can be helpful for debugging. As soon as W or S are let go autonomous takes over again.
- Pressing the spacebar in training mode toggles on and off cruise control (effectively presses W for you).

## Run simulation
After you have installed the simulator set it in `autonomous mode`. Then open your terminal an type:

```sh
git clone https://github.com/laygond/Behavioral-Cloning.git
cd Behavioral-Cloning
python drive.py model.h5
```
The above command will load the trained model and use the model to make steering angle predictions on individual images in real-time

#### Saving a video of your autonomous simulation
By running the following commands it will save your simulation as single frame images in directory `output_run` (if the directory already exists, it'll be overwritten). Then based on images found in the `output_run` directory a video will be created with the name of the directory followed by `'.mp4'`, so, in this case the video will be `output_run.mp4`. Finally, and optionally, remove the directory `output_run` which is no longer needed.

```sh
python drive.py model.h5 output_run
python video.py output_run
rm -rf output_run
```
Alternatively, one can specify the FPS (frames per second) of the video. The default FPS is 60 if not specified:
```sh
python video.py output_run--fps 48
```


* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)
This README file describes how to output the video in the "Details About Files In This Directory" section.



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md
## Dataset
- In training mode you can stop and play the recording as many times as you want and all collected data will be appended to the folder you have specified. But if you close and reopen the session by setting the same folder path from before then it will be overwritten.

If everything went correctly for recording data, you should see the following in the directory you selected. In my case I placed it in `MYDATA` at the same level directory as this repo as shown in `Directory Structure`
- IMG folder - this folder contains all the frames of your driving.
- driving_log.csv - each row in this sheet correlates your image with the steering angle, throttle, brake, and speed of your car. You'll mainly be using the steering angle.

![alt text][image1]
For this project we will only use the center camera images highlighted in gray and the steering angles. 

#### Strategies for Collecting Data
Collecting data correctly will ensure a successful model:

- the car should stay in the center of the road as much as possible
- if the car veers off to the side, it should recover back to center
- driving counter-clockwise can help the model generalize
- flipping the images is a quick way to augment the data
- collecting data from the second track can also help generalize the model
- we want to avoid overfitting or underfitting when training the model
- knowing when to stop collecting more data
- using the left and right cameraas if they were in the center by applying a small shift in the angle

#### Training and Validating Your Network
In order to validate your network, you'll want to compare model performance on the training set and a validation set. The validation set should contain image and steering data that was not used for training. A rule of thumb could be to use 80% of your data for training and 20% for validation or 70% and 30%. Be sure to randomly shuffle the data before splitting into training and validation sets.

If model predictions are poor on both the training and validation set (for example, mean squared error is high on both), then this is evidence of underfitting. Possible solutions could be to

increase the number of epochs
add more convolutions to the network.
When the model predicts well on the training set but poorly on the validation set (for example, low mean squared error for training set, high mean squared error for validation set), this is evidence of overfitting. If the model is overfitting, a few ideas could be to

use dropout or pooling layers
use fewer convolution or fewer fully connected layers
collect more data or further augment the data set


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.





### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
