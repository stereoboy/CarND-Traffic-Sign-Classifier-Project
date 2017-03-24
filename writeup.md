# **Traffic Sign Recognition**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.png         "Visualization"
[image2]: ./augmented_data.png        "Augmented Data"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./1.jpg "Traffic Sign 1"
[image5]: ./2.jpg "Traffic Sign 2"
[image6]: ./3.jpg "Traffic Sign 3"
[image7]: ./4.jpg "Traffic Sign 4"
[image8]: ./5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 86,430
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing how the label distribution is not uniform.

So we need to balance data with respect to its label by duplicate images of rare label

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 4th code cell of the IPython notebook.

As a first step, I balance label distribution by duplicating rare label images. Data size is enlarged from 34,799 to 86,430.

After balancing step, I 'just' define image data augmentation functions for later step.

Unfortunately I cannot make augmented data before training because of memory limit.

Data augmentation is needed essentially for recognizing badly cropped images and noisy data.

So I preprocess data every epoch in training step. My implementation is described as follows

* Geometrical manipulations are performed: scale-up and rotation and traslation
* And then I modify brightness. For this process, conversion from RGB to HSV and recovering are needed.
  * I decided not to convert the images to grayscale because some signs look similar without color information.
* In a final step, I normalize by subtracting mean and dividing by standard deviation.

Here are balanced dataset histogram and an example of several augmented traffic sign images

![alt text][image2]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

To cross validate my model, I use default setup. The data is already split into training set, validation set and test set.

My final training set had 86,430 number of images. My validation set and test set had 4,410 and 12,630 number of images.

The 6th code cell of the IPython notebook contains the code for augmenting the data set by calling augmentation function defined above. 

I generate additional data every epoch because of memory problem.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 5th cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 5x5x48 	|
| RELU					|												|
| Fully connected		| outputs 120        									|
| Fully connected		| outputs 84   									|
| Fully connected		| outputs 43   									|
| Softmax				| 43        									|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I use AdamOptimizer with learning rate = 0.001. It works well.
Batch size is 128, and epochs = 35. I need a lot of iterations because I apply data augmentation function on original training data every epoch. As a result 35 epochs means 30 augmentation process x 1 epoch.
The code for training the model is located in the 8th cell of the ipython notebook.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 8th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.951
* test set accuracy of 0.930

I just upgrade LeNet by adding one 1x1 convolution layer and expading filter size.
I thought that basically LeNet model is enough for small 32x32 images. so I just expand filter size of convolution layers for detecting various edge types. And add one 1x1 convolution layer and expand output size of fully connected layers to make the model a bit more complicated.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fifth image might be difficult to classify because sign is a bit shifted below.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 9th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 										|
| Children crossing					| Children crossing											|
| Stop	      		| Stop					 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. Since new images are more clear than training set, So result is good.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10 th cell of the Ipython notebook.

For the first image, the model is sure that this is a 'No entry' sign (probability of 1.0). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0|No entry|
|1.87334e-19|Stop|
|1.63233e-21|Priority road|
|6.23938e-28|No passing|
|1.22886e-28|No passing for vehicles over 3.5 metric tons|

For the second image is a "Speed limit (60km/h)".

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0|Speed limit (60km/h)|
|2.63009e-07|Speed limit (20km/h)|
|4.51405e-08|Speed limit (30km/h)|
|1.96506e-09|Speed limit (50km/h)|
|3.52795e-10|Speed limit (80km/h)|

The third image is a 'Children crossing'.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0|Children crossing|
|4.61107e-18|Beware of ice/snow|
|2.02358e-18|Right-of-way at the next intersection|
|5.64886e-20|Bicycles crossing|
|6.4604e-25|Dangerous curve to the right|

The forth image is 'Stop'.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0|Stop|
|4.43008e-18|Priority road|
|4.78567e-20|Yield|
|3.44521e-20|No entry|
|1.10641e-20|Speed limit (30km/h)|

The fifth image is 'Speed limit (30km/h)'

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.0|Speed limit (30km/h)|
|2.81762e-08|Speed limit (70km/h)|
|9.19204e-12|Speed limit (20km/h)|
|5.47486e-18|Yield|
|2.36564e-19|Double curve|


