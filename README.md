# Lane Detection and Steering (only Image Processing)
 - This projects detects lane in the video and then calculates the steering angle needed for following the lane.
 - Author : Suyash Verma (suchiv2311@gmail.com)

### Implementation
 - The input file was sourced from here -> (https://youtu.be/6q5_A5wOwDM)
 - No advanced concepts were used like ML, DL, AI etc.
 - This pipeline uses Image Processing techniques like Image-Thresholding, Morphological Transformations, Histogram analysis, Hough-Line Transform, Masking etc.
 - It also applies some mathematical and statistical functions to generalize the output.
 - Then it computes the center of the lane and finding the angle that needs to be steered. (Note: This is not absolute reading, but a intermediary value that can be used in control algorithms like PID.
 - This all is built using Python Libraries namely: OpenCV, Numpy, statistics, math, etc.

### Sample Output:
 - This is a frame from the output of the project.
 ![one frame of output](https://raw.githubusercontent.com/SuyashVerma2311/Lane-Detection-and-Steering/main/Pipeline/Sample_image.jpg?token=ANIF34FSQCYIVE7DZYSKBHTAUP4Q2)
