# SELF-DRIVING CAR USING UDACITY’S CAR SIMULATOR ENVIRONMENT AND TRAINED BY DEEP NEURAL NETWORKS 


## Table of Contents

## Overview

In this project, we use deep neural networks and convolutional neural networks to clone driving behavior. The model is trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle. The autonomous vehicle is provided as a simulator. Image data and steering angles are used to train a neural network and drive the simulation car autonomously around the track.
This project started with training the models and tweaking parameters to get the best performance on the track 
The use of CNN for getting the spatial features and RNN for the temporal features in the image dataset makes this combination a great fit for building fast and lesser computation required neural networks. Substituting recurrent layers for pooling layers might reduce the loss of information and would be worth exploring in the future projects. 
It is interesting to find the use of combinations of real world dataset and simulator data to train these models. Then I can get the true nature of how a model can be trained in the simulator and generalized to the real world or vice versa. There are many experimental implementations carried out in the field of self-driving cars and this project contributes towards a significant part of it. 
### Introduction

- Problem Definition
- Solution Approach
- Technologies Used
- Convolutional Neural Networks (CNN)
- Time-Distributed Layers

### Udacity Simulator and Dataset
### The Training Process
### Augmentation and image pre-processing
### Experimental configurations
### Network architectures
### Results

- Value loss or Accuracy
- Why We Use ELU Over RELU


### The Connection Part
### Files
### Overview
### References

## Introduction 
Self-drivi cars has become a trending subject with a significant improvement in the technologies in the last decade. The project purpose is to train a neural network to drive an autonomous car agent on the tracks of Udacity’s Car Simulator environment. Udacity has released the simulator as an open source software and  enthusiasts have hosted a competition (challenge) to teach a car how to drive using only camera images and deep learning. Driving a car in an autonomous manner requires learning to control steering angle, throttle and brakes. Behavioral cloning technique is used to mimic human driving behavior in the training mode on the track. That means a dataset is generated in the simulator by user driven car in training mode, and the deep neural network model then drives the car in autonomous mode. Ultimately, the car was able to run on Track 1 generalizing well. The project aims at reaching the same accuracy on real time data in the future.![6](https://user-images.githubusercontent.com/91852182/147298831-225740f9-6903-4570-8336-0c9f16676456.png)


### Problem Definition

Udacity released an open source simulator for self-driving cars to depict a real-time environment. The challenge is to mimic the driving behavior of a human on the simulator with the help of a model trained by deep neural networks. The concept is called Behavioral Cloning, to mimic how a human drives. The simulator contains two tracks and two modes, namely, training mode and autonomous mode. The dataset is generated from the simulator by the user, driving the car in training mode. This dataset is also known as the “good” driving data. This is followed by testing on the track, seeing how the deep learning model performs after being trained by that user data.

### Solution Approach
 ![1](https://user-images.githubusercontent.com/91852182/147298261-4d57a5c1-1fda-4654-9741-2f284e6d0479.png)
 
 The problem is solved in the following steps: 
 
- The simulator can be used to collect data by driving the car in the training mode using a joystick or keyboard, providing the so called “good-driving” behavior input data in form of a driving_log (.csv file) and a set of images. The simulator acts as a server and pipes these images and data log to the Python client. 
- The client (Python program) is the machine learning model built using Deep Neural Networks. These models are developed on Keras (a high-level API over Tensorflow). Keras provides sequential models to build a linear stack of network layers. Such models are used in the project to train over the datasets as the second step. Detailed description of CNN models experimented and used can be referred to in the chapter on network architectures. 
- Once the model is trained, it provides steering angles and throttle to drive in an autonomous mode to the server (simulator). 
- These modules, or inputs, are piped back to the server and are used to drive the car autonomously in the simulator and keep it from falling off the track.

### Technologies Used 

Technologies that are used in the implementation of this project and the motivation behind using these are described in this section.
 
TensorFlow: This an open-source library for dataflow programming. It is widely used for machine learning applications. It is also used as both a math library and for large computation. For this project Keras, a high-level API that uses TensorFlow as the backend is used. Keras facilitate in building the models easily as it more user friendly. 

Different libraries are available in Python that helps in machine learning projects. Several of those libraries have improved the performance of this project. Few of them are mentioned in this section. First, “Numpy” that provides with high-level math function collection to support multi-dimensional metrices and arrays. This is used for faster computations over the weights (gradients) in neural networks. Second, “scikit-learn” is a machine learning library for Python which features different algorithms and Machine Learning function packages. Another one is OpenCV (Open Source Computer Vision Library) which is designed for computational efficiency with focus on real-time applications. In this project, OpenCV is used for image preprocessing and augmentation techniques. 

The project makes use of Conda Environment which is an open source distribution for Python which simplifies package management and deployment. It is best for large scale data processing. The machine on which this project was built, is a personal computer. 

### Convolutional Neural Networks (CNN)

CNN is a type of feed-forward neural network computing system that can be used to learn from input data. Learning is accomplished by determining a set of weights or filter values that allow the network to model the behavior according to the training data. The desired output and the output generated by CNN initialized with random weights will be different. This difference (generated error) is backpropagated through the layers of CNN to adjust the weights of the neurons, which in turn reduces the error and allows us produce output closer to the desired one. 

CNN is good at capturing hierarchical and spatial data from images. It utilizes filters that look at regions of an input image with a defined window size and map it to some output. It then slides the window by some defined stride to other regions, covering the whole image. Each convolution filter layer thus captures the properties of this input image hierarchically in a series of subsequent layers, capturing the details like lines in image, then shapes, then whole objects in later layers. CNN can be a good fit to feed the images of a dataset and classify them into their respective classes. 

### Time-Distributed Layers

Another type of layers sometimes used in deep learning networks is a Time- distributed layer. Time-Distributed layers are provided in Keras as wrapper layers. Every temporal slice of an input is applied with this wrapper layer. The requirement for input is that to be at least three-dimensional, first index can be considered as temporal dimension. These Time-Distributed can be applied to a dense layer to each of the timesteps, independently or even used with Convolutional Layers. The way they can be written is also simple in Keras as shown in Figure 1 and Figure 2.

![2](https://user-images.githubusercontent.com/91852182/147298483-4f37a092-7e71-4ce6-9274-9a133d138a4c.png)

Fig. 1: TimeDistributed Dense layer

![3](https://user-images.githubusercontent.com/91852182/147298501-6459d968-a279-4140-9be3-2d3ea826d9f6.png)

Fig. 2: TimeDistributed Convolution layer

## Udacity Simulator and Dataset 

We will first download the [simulator](https://github.com/udacity/self-driving-car-sim) to start our behavioural training process. Udacity has built a simulator for self-driving cars and made it open source for the enthusiasts, so they can work on something close to a real-time environment. It is built on Unity, the video game development platform. The simulator consists of a configurable resolution and controls setting and is very user friendly. The graphics and input configurations can be changed according to user preference and machine configuration as shown in Figure 3. The user pushes the “Play!” button to enter the simulator user interface. You can enter the Controls tab to explore the keyboard controls, quite similar to a racing game which can be seen in Figure 4. 

![ 4](https://user-images.githubusercontent.com/91852182/147298708-de15ebc5-2482-42f8-b2a2-8d3c59fceff4.png)

Fig. 3: Configuration screen                                                                    

![5](https://user-images.githubusercontent.com/91852182/147298712-944e2c2d-e01d-459b-8a7d-3c5471bea179.png)

Fig. 4: Controls Configuration

The first actual screen of the simulator can be seen in Figure 5 and its components are discussed below. The simulator involves two tracks. One of them can be considered as simple and another one as complex that can be evident in the screenshots attached in Figure 6 and Figure 7. The word “simple” here just means that it has fewer curvy tracks and is easier to drive on, refer Figure 6. The “complex” track has steep elevations, sharp turns, shadowed environment, and is tough to drive on, even by a user doing it manually. Please refer Figure 6. There are two modes for driving the car in the simulator: (1) Training mode and (2) Autonomous mode. The training mode gives you the option of recording your run and capturing the training dataset. The small red sign at the top right of the screen in the Figure 6 and 7 depicts the car is being driven in training mode. The autonomous mode can be used to test the models to see if it can drive on the track without human intervention. Also, if you try to press the controls to get the car back on track, it will immediately notify you that it shifted to manual controls. The mode screenshot can be as seen in Figure 8. Once we have mastered how the car driven controls in simulator using keyboard keys, then we get started with record button to collect data. We will save the data from it in a specified folder as you can see below.

![6](https://user-images.githubusercontent.com/91852182/147298837-17eecb80-0a3f-4edb-a5f3-050a318f66e0.png)

<img alt="7" src="https://user-images.githubusercontent.com/91852182/147298975-e05dc738-2fb7-4dca-a28d-9756285d94cc.png">

The simulator’s feature to create your own dataset of images makes it easy to work on the problem. Some reasons why this feature is useful are as follows: 

- The simulator has built the driving features in such a way that it simulates that there are three cameras on the car. The three cameras are in the center, right and left on the front of the car, which captures continuously when we record in the training mode. 
- The stream of images is captured, and we can set the location on the disk for saving the data after pushing the record button. The image set are labelled in a sophisticated manner with a prefix of center, left, or right indicating from which camera the image has been captured. 
- Along with the image dataset, it also generates a datalog.csv file. This file contains the image paths with corresponding steering angle, throttle, brakes, and speed of the car at that instance. 

A few images from the dataset are shown below .

![Driving Sample](https://github.com/vivekviplov/SelfDrivingCar/blob/main/left_2025_06_30_22_33_32_036.jpg)
![Driving Sample](https://github.com/vivekviplov/SelfDrivingCar/blob/main/center_2025_06_30_22_28_47_412.jpg)
![Driving Sample](https://github.com/vivekviplov/SelfDrivingCar/blob/main/right_2025_06_30_22_37_52_147.jpg)


A sample of driving_log.csv file is shown in Figure 9.

* *Column 1, 2, 3:*  contains paths to the dataset images of center, right and left respectively Column 4: contains the steering angle 
* *Column 4:*  value as 0 depicts straight, positive value is right turn and negative value is left turn. 
* *Column 5:*  contains the throttle or acceleration at that instance 
* *Column 6:*  contains the brakes or deceleration at that instance 
* *Column 7:*  contains the speed of the vehicle 

![9](https://github.com/vivekviplov/SelfDrivingCar/blob/main/Screenshot%202025-07-06%20170107.png)
**Fig. 9: Driving_log_csv**


## Augmentation and image pre-processing
The biggest challenge was generalizing the behavior of the car on Track_2 which it was never trained for. In a real-life situation, we can never train a self-driving car model for every track possible, as the data will be too huge to process. Also, it is not possible to gather the dataset for all the weather conditions and roads. Thus, there is a need to come up with an idea of generalizing the behavior on different tracks. This problem is solved using image preprocessing and augmentation techniques.

## Network architectures

The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.It is a deep convolution network which works well with supervised image classification / regression problems. As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model.

- I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
- I've added an additional dropout layer to avoid overfitting after the convolution layers.
- I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: ELU
- Fully connected: neurons: 50, activation: ELU
- Fully connected: neurons: 10, activation: ELU
- Fully connected: neurons: 1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle. However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction. Overall, the model is very functional to clone the given steering behavior. The below is a model structure output from the Keras which gives more details on the shapes and the number of parameters.

We will design our Model architecture. We have to classify the traffic signs too that's why we need to shift from Lenet 5 model to NVDIA model. With behavioural cloning, our dataset is much more complex then any dataset we have used. We are dealing with images that have (200,66) dimensions. Our current datset has 5386 images to train with but MNSIT has around 60,000 images to train with. Our behavioural cloning code has simply has to return appropriate steering angle which is a regression type example. For these things, we need a more advanced model which is provided by nvdia and known as nvdia model.

For defining the model architecture, we need to define the model object. Normalization state can be skipped as we have already normalized it. We will add the convolution layer. As compared to the model, we will organize accordingly. The Nvdia model uses 24 filters in the layer along with a kernel of size 5,5. We will introduce sub sampling. The function reflects to stride length of the kernel as it processes through an image, we have large images. Horizontal movement with 2 pixels at a time, similarly vertical movement to 2 pixels at a time. As this is the first layer, we have to define input shape of the model too i.e., (66,200,3) and the last function is an activation function that is “elu”.

Revisting the model, we see that our second layer has 36 filters with kernel size (5,5) same subsampling option with stride length of (2,2) and conclude this layer with activation ‘elu’.

According to Nvdia model, it shows we have 3 more layers in the convolutional neural network. With 48 filters, with 64 filters (3,3) kernel 64 filters (3,3) kernel Dimensions have been reduced significantly so for that we will remove subsampling from 4th and 5th layer.

Next we add a flatten layer. We will take the output array from previous convolution neural network to convert it into a one dimensional array so that it can be fed to fully connected layer to follow.

Our last convolution layer outputs an array shape of (1,18) by 64.

We end the architecture of Nvdia model with a dense layer containing a single output node which will output the predicted steering angle for our self driving car. Now we will use model.compile() to compile our architecture as this is a regression type example the metrics that we will be using will be mean squared error and optimize as Adam. We will be using relatively a low learning rate that it can help on accuracy. We will use dropout layer to avoid overfitting the data. Dropout Layer sets the input of random fraction of nodes to “0” during each update. During this, we will generate the training data as it is forced to use a variety of combination of nodes to learn from the same data. We will have to separate the convolution layer from fully connected layer with a factor of 0.5 is added so it converts 50 percent of the input to 0. We Will define the model by calling the nvdia model itself. Now we will have the model training process.To define training parameters, we will use model.fit(), we will import our training data X_Train, training data -&gt;y_train, we have less data on the datasets we will require more epochs to be effective. We will use validation data and then use Batch size.


## Results

The following results were observed for described architectures. I had to come up with two different performance metrics. 
- Value loss or Accuracy (computed during training phase) 
- Generalization on Track 1 (drive performance)

### Why We Use ELU Over RELU

We can have dead relu this is when a node in neural network essentially dies and only feeds a value of zero to nodes which follows it. We will change from relu to elu. Elu function has always a chance to recover and fix it errors means it is in a process of learning and contributing to the model. We will plot the model and then save it accordingly in h5 format for a keras file.
