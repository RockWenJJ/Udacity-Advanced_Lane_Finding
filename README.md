# **Finding Lane Lines on the Road** 

![output result](./tmp_output/example.png)

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that 
show us where the lanes are act as our constant reference for where to steer the 
vehicle.  Naturally, one of the first things we would like to do in developing a 
self-driving car is to automatically detect lane lines using an algorithm.

This project detect lane lines in images using Python and OpenCV. This project was 
successfully tested in Python3.6 environment. The required packages include:
```python
matplotlib
python-opencv
numpy
os
math
```

How to use this project
---
To have a brief understanding of the detection pipeline, you should first take a
 look at [writeup.md](writeup.md), which describes the pipeline as well as the 
 shortcomings and possible improvements.
 
 To see the detection results, you could take a look at folders `test_images_output`  and `test_videos_output`,
 which contain the output images and videos respectively. If you are interested in the 
 temporary outputs, you could take a look at folder `tmp_output`.
 
 The detection pipeline is written in `submission.ipynb`, which is a ipython-notebook file. You can
 follow the directions step by step in that notebook file to see the details of this method. 
 `submission.html` also shows you the code details, while the only difference is that you cann't run 
 the code.
 
 This project has been pushed to [github](https://github.com/RockWenJJ/Udacity-Advanced_Lane_Finding).
 