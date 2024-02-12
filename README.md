# ComputerVision - Image Processing and Computer Vision Project

## Project Description:

This project, developed within the field of Image Processing and Computer Vision, was created using the Python programming language and the libraries OpenCV and NumPy. The project aims to detect Aruco markers, coins, and Lego bricks in images or videos.

## Implemented Features:

1. **Image or Video Processing:**
   - The program can process either an image from a file (`sys.argv[1]`) or video capture from a camera.
   - The camera dimensions are set to 640x480 pixels (`CAM_WIDTH`, `CAM_HEIGHT`).

2. **Paper Detection:**
   - When an image is specified, the program attempts to recognize an A4 paper in the image.
   - Edges are detected, contours are found, and the largest rectangle (presumably the A4 paper) is highlighted.

3. **Coin Detection:**
   - Circles in the image are found using the Hough transformation and marked.
   - The diameter of the detected circles is measured, and based on known diameters, coin types are assigned.
   - The recognized coins are marked on the image, their coin type is output, and the total sum in euros is calculated.

4. **Aruco Marker Detection:**
   - If an image is used as input, Aruco markers are detected and highlighted on the image.
   - The width of the Aruco markers in pixels and their width in millimeters are calculated.

5. **Perspective Distortion Correction:**
   - If an A4 paper is detected, a perspective distortion correction of the image is performed to obtain a clear view of the paper.

6. **Lego Brick Detection:**
   - The program performs color segmentation in the HSV color space to detect gray, orange, and yellow Lego bricks.
   - Contours are found and marked on the image.
   - The number of detected Lego bricks of each color is output.

## Code Structure:

- The code includes blocks for image and video processing, depending on the input parameters.
- There are blocks for paper detection, coin detection, Aruco marker detection, perspective distortion correction, and Lego brick detection.
- Functions like `count_brick_contours` count the number of detected Lego bricks.
- The code employs various techniques such as edge detection, Hough transformation, HSV color space segmentation, and perspective transformation.

In summary, the project provides a practical application of image processing and computer vision techniques to recognize and analyze various objects in images or videos.
