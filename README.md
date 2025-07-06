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


