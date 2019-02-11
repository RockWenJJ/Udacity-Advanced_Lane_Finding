## Advanced Lane Finding Project
---

### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### **Camera Calibration**

The code for this step is contained in the second code cell of the IPython notebook located in "submission.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  Then save the camera calibration parameters to `camCalib.pkl`. Finally, this distortion correction is applied to the test image with the `cv2.undistort()` function and I obtained this result: 

![undistorted chessboard][./tmp_output/undistort_chessboard.png]

### Pipeline (single images)

1. **Apply distortion correction to input image**

Firstly, I apply the distortion correction to one of the test images like this one:

![undistorted image][./tmp_output/undistort.png]

2. **Create thresholded binary images using color and gradient selection**

I used a combination of color and gradient thresholds to generate a binary image. The code of creating binary image is located in Cell 6 and 7 of `submission.ipnb`. Here's an example of my output for this step:

![undistorted image][./tmp_output/binary_image.png]

3. **Apply perspective transform**

The core of perspective transform is properly choosing the source corners and destination corners. With reference to `straight_lines1.jpg`, the source and destination points in the are set in the following manner:

```python
offset = 100
src_corners = np.array([[231, 693],
                       [530, 493],
                       [756, 493],
                       [1084, 693]], dtype=np.float32)

dst_corners = np.array([[340, 693],
                       [340, offset],
                       [940, offset],
                       [940, 693]], dtype=np.float32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 231, 693      | 340, 693      | 
| 530, 493      | 340, 100      |
| 756, 493      | 940, 100      |
| 1084, 693     | 940, 693      |

The code is shown in the code cell 8 in `submission.ipynb`.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped image][./tmp_output/warped.png]
![warped binary][./tmp_output/warped_binary.png]

4. **Detect lane-line pixels and fit with a polynomial**

The lane-line pixels are detected using window convolutional method and the pixels are fitted with a quadratic polynomial function. The code is shown in the code cell 11 through 12 in `submission.ipynb`.

5. **Calculate radius of curvature and the position of the vehicle with respect to the lane center**

If the lane line in pixels is described as ```x = a*(y**2)+b*y+c```, and `mx`/`my` are the scale for the x and y dimension, then the scaled lane line in meters is ```x = mx/(my**2)*a*(y**2) + (mx/my)*b*y +c```. So the lane line curvature in meters is computed as follows:

```python
ar = mx/(my**2)*a
br = (mx/my)*b
ploty_cr = ploty * ym_per_pix
y_eval = np.max(ploty_cr)

curv = (1+2*(ar*y_eval+br)**2)**(3/2)/np.abs(2*ar)
```

The position of the vehicle to the center is computed with the assumption that the center of the warped image is the lane center. So the position is computed as follows:

```python
computed_center = (leftx_eval_pix + rightx_eval_pix)/2
detour_real = abs(computed_center - 640) * xm_per_pix

```

The code is located in code cell 12 of `submission.ipynb`.

6. **Output detecting results**

The overall detecting methods are combined into function `detectine_pipeline()` in code cell 14 of `submission.ipynb`. Here is an example of my result on a test image:

![output result][./tmp_output/example.png]

---

### Video Output

Here's the [result](./tmp_output/project_res.mp4) of video lane line detection.

---

### Discussion

Currently, the method is applicable in situations when the weather is good, the lane lines are clear, the turn is not too sharp and so forth. While in some extreme situations, like the situations in  `challenge_video.mp4` and `harder_challenge_video.mp4`, this method doesn't perform well.

Therefore, this method could be improved by the following approaches:
* Detecting lane lines with the state-of-art Convolutional Neural Network in order to deal with extreme situations.
* Use some tracking method in order to deal with video streams when lane lines are lost in some frames.

Moreover, the visual estimation of curvature and position could be erroneous if there is a big slope in the forward road. So maybe a better estimation could be achieved with the help of other sensors and maps.