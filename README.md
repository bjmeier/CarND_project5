## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---
[//]: # (Image References)
[image1]: ./report/car_notcar.png

[image2]: ./report/Hog_car_gray.png
[image3]: ./report/Hog_car_Y.png
[image4]: ./report/Hog_car_U.png
[image5]: ./report/Hog_car_V.png

[image6]: ./report/Hog_nc_gray.png
[image7]: ./report/Hog_nc_Y.png
[image8]: ./report/Hog_nc_U.png
[image9]: ./report/Hog_nc_V.png

[image10]: ./report/Window1.png
[image11]: ./report/Window2.png
[image12]: ./report/Window3.png
[image13]: ./report/Window4.png
[image14]: ./report/Window5.png
[image15]: ./report/Window6.png


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first seven code cells of the IPython notebook `CarND_Project5_vehicle_Detection.ipynb`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]


#### 2. Explain how you settled on your final choice of HOG parameters.


I used the code in lesson 28, Color Classify' to determine the color space as well as the and size of the spacial and color histogram features.

First, I determined the colorspace used for the spacial features and color histogram features by investigating the random states of 0, 10, 20, 30 and 40 while keeping a 32x32 pixel size a 32 bin histogram. 

Colorspace|Avg Accuracy
--- | ---
**RGB** | **0.982**
HSV | 0.976
LUV | 0.977
HLS | 0.970
YUV | 0.972

Based on the accuracy given above, `RGB` was selected.  Next, I investigated several pixel sizes for the same random states.

Pixel Size | Bin Size | Avg Accuracy
--- | --- | ----
16x16 | 32 | 0.985
32x32 | 32 | 0.982
** 8x8 ** | ** 32 ** | ** 0.990 **
4x4 | 32 | 0.986

Based on the accuracy given above, an 8x8 pixel size was selected. Histogram bin size was then investigated.

Pixel Size | Bin Size | Avg Accuracy
--- | --- | ----
8x8 | 16 | 0.973
** 8x8 ** | ** 32 ** | ** 0.990 **
8x8 | 64 | 0.988

Based on the accuracy given above, a 32 bin histogram was selected.

For the HOG features, I used the same ones referenced in the [original paper](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) by Dalal and Triggs. These are 8x8 pixels per bin, 8x8 bins per image, 9 orientations 'L2-Hys' normalization. 

To determine the colorspace, I used the code in lesson 34, Search and Classify. I used the above pixel sizes, histogram bins and the `RGB` color space. Using 0, 10, 20, 30 and 40 random states, the average accuracies are given in the below table.

HOG Colorspace | Avg Accuracy
--- | ---
RGB | 0.963
HSV | 0.982
LUV | 0.984
HLS | 0.983
** YUV ** | ** 0.988 ** 
YCrCb | 0.986

Based on the above table, the `YUV` colorspace was selected.  Due to a difficulty in detecting white cars, the `Gray` colorspace was added.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Linear SVM using the KITTI and all GTI car images as well as the Extras and GTI not car images. The number of car images is 8,792.  The number of not car images is 8,968. Prior to fitting, the data was normalized using the `sklearn.preprocessing Standard Scaler` function.

Using a 80% training, 20% test split, the below `C` values were investigated for one random state.

C | Accuracy | Training Time (s)
--- | --- | ---
0.0001 | 0.929 | 52.0
0.001 | 0.930 | 32.2
** 0.01 ** | ** 0.927 ** | ** 19.8 **
0.1 | 0.916 | 16.5
1 | 0.915 | 16.1
10 | 0.913 | 16.8
100 | 0.914 | 16.3
1,000 | 0.913 | 16.1
10,000 | 0.913 | 15.9

Because little accruacy was gained by using a lower `C` value, the same `C` value used in the  [original paper](http://vc.cs.nthu.edu.tw/home/paper/codfiles/hkchiu/201205170946/Histograms%20of%20Oriented%20Gradients%20for%20Human%20Detection.pdf) was selected.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used 64x64 and 128x128 windows. I used an overlap of 25%. I searched the right side of the image.  The ystart value is 392 pixels.  The 64x64 windows included three verticle steps and the 128x128 windows searched four vertical steps. Images are searched in the code blocks under the Add windows... and Find cars.. blocks.  These sizes and search areas did a good job of overing the area of interest.  The 25% overlap provided a good balance between speed and search precision.  Because I am using 8 x8 hog blocks, moving two at a time aided ease of implementation and performance speed.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on the above two scales using YUV plus gray 4-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

Descriptions of thresholds and filter used to prevent false positives are given below.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a link to my video result.  This video was generated a a rate slightly greater than 3 frames per second on a laptop c an Intel 7th Generation i5 core.

 [![link to my video result](http://img.youtube.com/vi/5PX58LZ7aZQ/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_5PX58LZ7aZQ)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap.  The heatmap is generted fromt square of the distance generated by `svc.decision_function`.  The heatmap is then clipped at 2.5.  This heatmap is then fed into an Exponential moving average (EMA) filter with a smoothing factor of 0.05. The result of the averaged heatmap then is subjected to a minimum threshold of 1.5. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Video showing window inputs

To show how this works as a function of time, I created a video.  This video shows the detected cars in blue.  Black boarders indicate uncertain (close to the decision line).  Blue lines indicated more certain car detections.  Detected cars are shown by the red border.  A link to this video is given below. 

 [![link to my video result](http://img.youtube.com/vi/vHNPEkrmA5Q/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_vHNPEkrmA5Q)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult issue was dealing with shadows.  To counteract this, a low pass filter was required. The elimated any panic stops, but delayed detection.  To counter this, I suggest using a more robust deep learning  based detection method. 

