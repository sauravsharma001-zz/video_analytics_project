# Camera Resectioning
Given:
1) Color and depth frames - an initial and a final frame of the sequence - obtained from Kinect, along with time instants of capture.
2) Camera intrinsic matrix for RGB camera, 
3) Inverse of camera intrinsic matrix of the depth camera and rotation matrix between RGB and depth camera. 

The main goal of the assignment was:
1)	Using these matrices, find a corresponding color for each pixel in the depth image. This will generate a colorized depth image.  In the colorized depth image, draw bounding boxes around the two balls. Colorized depth image (and draw boinding boxes) to be generated for both initial and final set of frames.
2)	Using the initial and final position of two balls in Color and depth frames and calibration parameters, relative velocity of balls to be calculated in milimeters/seconds.


# TEMOC Detection
Given a set of images as training data, program was trained to detect TEMOC (UTD Mascot) in real-time video using KNN Classifier.

# Final Project
Tasks needed to accomplish:
1.	Work on real-time video (i.e., not captured video)
2.	Video should have few persons (say 2 to 4) moving around and at least one of them should be wearing UTD logo T-shirts. 
3.	You can choose any T-shirt with a logo that is big and easy to recognize. And track the person wearing T-shirt with that Logo.
4.	Your task is to detect and track only the person wearing that logo T-shirt, i.e., put a bounding box on the entire person wearing the T-shirt (not just the T-shirt alone).
5.	Additionally, mark the tracked person’s face and eyes with different colored bounding boxes.
6.	A separate task for the project is to determine the height of the person in feet and inches. To do this you can use an object of known width and height to act as a calibration parameter. You need to track that object and the person, find the ratio and obtain the person’s real height. In order to do this, the person needs to be standing next to the object, at the same depth location.
7.	Person, face, eye, logo and object detection can be done using any OpenCV strategy – no restrictions.