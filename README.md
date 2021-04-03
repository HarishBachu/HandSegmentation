# HandSegmentation

This project allows you to create your own Hand Dataset using your webcam, for your own mini Gesture Detection Project, or any related applications.
The Segmentation is sensitive to lighting conditions, and moving objects in the background, work is being done to make the algorithm more robust to these. 

The algorithm uses both motion detection (done by Background Subtraction), and skin detection (by capturing pixels of intensities in some pre-defined range) to isolate skin areas in the webcam feed, frame by frame. The largest contour is extracted from this result, which is used to generate the final mask. In almost all cases, the largest contour turns out to be the Hand itself. 

## Motivation ##

This project was motivated by one of my other projects regarding gesture recognition. Not many good datasets are available online for gesture recognition. Finding reliable datasets online, and downloading them was time and data intensive. Ultimately, I had better performance with my own generated dataset, than online datasets. So I decided to make a tool for other Computer Vision enthusiasts to get acquainted with the field. 

## Prerequisites: ##

To run this tool, you need the following libraries:

* numpy
* opencv
* imtils
* math
* os
* pymsgbox
* tqdm

To install these, simply run the requirements.txt file in this repository (after cloning onto your system) using pip by running the following command.
```
pip install -r requirements.txt
```

Or you can install them separately.

## How to use: ##
 
![](example.gif)

1. Clone the repository either by __downloading it as a zip__ and unzipping the folder, or by using __git clone__ into your preferred directory:
```
git clone https://github.com/HarishBachu/HandSegmentation.git
```

2. Navigate to the cloned repository and run the python file named "main.py"
```
python main.py
```

3. You will see a field to enter __root directory__. Enter your preferred root directory address. This is where you want to create your image dataset. 
If your directory location does not exist, a new directory with that name will be created and used. 
```
Enter Root Directory: (Enter where you want to create your dataset here) 
                      (Eg: /home/<username> for a linux system)
```

4. Next, you will be asked to enter the __number of images__ you want to generate. Enter the required number of images.
```
Enter number of images for your Dataset: (Enter how many images you want here)
```

5. After entering the number of images, if the packages were installed properly, you should see 2 windows, showing webcam feed. One window will show a normal feed, and the other will show a blurred version. The blurring is done to facitilate the operations done on the image. 

   Find a location with proper lighting conditions for the algorithm to work properly. Make sure there are no moving objects in the background. Once this is done, press the **_'b'_** key on the keyboard to start the segmentation process. You should see a variety of popup windows, each showing some operation on the frame. 

   The final hand is given by the window named "Final Hand", and the mask for this is given in the window named "Final Mask". 

6. Once you are ready to start generating your dataset, press the **_'s'_** key on your keyboard. The program will continuously save the processed images in their respective subdirectories, inside the root directory. Once the process is done, a popup window will appear, alreting you that the process is complete.

7. If you wish to quit the program in between, press the **_'q'_** key.

## Future Modifications ##

1. The current method works well, but it requires the room to be well illuminated for accurate segmentation. Otherwise, the results are extremely noisy, and sometimes, it may crop out completely wrong areas of the image. Moreover, we operate under the assumption that the hand is the largest contour, which is not always the case. To overcome this, a novel CNN based architecture can be implemented to extract the required regions to a much better accuracy

2. The current version requires the download of the whole repository, and to run the code manually. However, some (if not most) users may prefer a more Graphical Approach. To remedy this, the whole program can be deployed as a web application (preferably on Heroku). I am currently familiarising myself with concepts of WebDev, and am looking to actively work on this aspect of the project. 
