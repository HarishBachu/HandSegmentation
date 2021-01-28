# HandSegmentation

This Project allows you to create your own Hand Dataset using your webcam, for your own mini Gesture Detection Project, or any related applications.
The Segmentation is sensitive to lighting conditions, and moving objects in the background, work is being done to make the algorithm more robust to these. 

The algorithm uses both motion detection (done by Background Subtraction), and skin detection (by capturing pixels of intensities in some pre-defined range) to isolate skin areas in the webcam feed, frame by frame. The largest contour is extracted from this result, which is used to generate the final mask. In almost all cases, the largest contour turns out to be the Hand itself. 

## Prerequisites: ##

To run this tool, you need the following libraries:

* numpy
* opencv
* imtils
* math
* os
* pymsgbox
