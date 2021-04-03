import numpy as np
import cv2
import os
from tqdm import tqdm
from Segmentor_App import segmentor
import pandas
import pymsgbox as msg

#Assigns Flags for each operation on the frame
def assignFlags():
    flagDict = {
        "flagMotion": False,    
        "flagSkin": False,
        "flagContour": False,
        "flagSkeleton": False
    }
    return flagDict

#Defines the root directory
#If the directory does not exist, the function creates one
def assignRootDir():
    root_dir = input('Enter Root Directory: ')
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    return root_dir

#Creates sub directories in the root directory, and returns their paths
def makeDirectories(root_dir, dir_list):
    dir_paths = []
    for folder_name in tqdm(dir_list, ascii = True, desc = "Creating Directories", ncols = 100):
        dir_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        dir_paths.append(dir_path)
    return dir_paths

def nothing(x):
    pass

if __name__ == "__main__":

    root_dir = assignRootDir()          #Creates Root directory
    dir_list = ['frame', 'motionMask', 'motion', 'skinMask', 'skin', 'finalMask', 'finalHand', 'skeleton']      #Sub Directories names
    
    datapaths = makeDirectories(root_dir, dir_list)     #paths to subdirectories
    # print(datapaths)
    cap = cv2.VideoCapture(0)       #Starts reading data from webcam
    
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    boundingBox = {'cx': width//2, 'cy': height//2}
    back = np.zeros((height, width, 3))

    returnDict = {}                 #Defines empty dictionary for processed images
    flags = assignFlags()           #Assigns processing flags

    segment = segmentor()           
    bgSubthreshold = 20
    bgSubtractorLr = 0
    flagSave = False                #Assigns Flag for saving the dataset
    
    num_imgs = int(input("Enter number of images for your Dataset: "))
    img_num = 0                     #Img index count starts from 0

    cv2.namedWindow('Box')
    cv2.createTrackbar('Width', 'Box', 250, width, nothing)
    cv2.createTrackbar('Height', 'Box', 250, height, nothing)

    boxWidth = 250
    boxHeight = 250
    while cap.isOpened():
            
        # boxWidth = cv2.getTrackbarPos('Width', 'Box')
        # boxHeight = cv2.getTrackbarPos('Height', 'Box')

        _, frame = cap.read()           #Data is read frame by frame
        box = frame.copy()
        blur = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        drawing = np.zeros((height, width, 3), dtype = np.uint8)
        finalMask = np.zeros((height, width), dtype = np.uint8)

        startCoord = (max(0, boundingBox["cx"] - boxWidth//2), max(0, boundingBox["cy"] - boxHeight//2))
        endCoord = (min(width, boundingBox["cx"] + boxWidth//2), min(height, boundingBox["cy"] + boxHeight//2))

        cv2.rectangle(box, startCoord, endCoord, 0xFF0000, 1)

        returnDict["blur"] = blur
        returnDict["frame"] = frame
        returnDict["drawing"] = drawing
        returnDict["finalMask"] = finalMask

        cv2.imshow('Frame', frame)
        cv2.imshow('Blurred', blur)
        cv2.imshow('Box', box)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):             #If the 'b' key is pressed, all the processing flags become True, and the segmentation starts
            bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, bgSubthreshold)
            back = frame
            keys = flags.keys()
            for i in keys:
                flags[i] = True

        #Flag for motion detection
        if flags["flagMotion"] == True: 
            returnDict = segment.detect_motion(bgSubtractor, bgSubtractorLr, returnDict)
            cv2.imshow('Motion Mask', returnDict["motionMask"])
            cv2.imshow('Motion', returnDict["motion"])

        #Flag for skin detection
        if flags["flagSkin"] == True:
            returnDict = segment.detect_skin(returnDict)
            cv2.imshow('Skin Mask', returnDict["skinMask"])
            cv2.imshow('Skin', returnDict["skin"])

        #Flag for contour extraction
        if flags["flagContour"] == True:
            returnDict, boundingBox = segment.get_contour(returnDict, boundingBox)
            cv2.imshow('Final Mask', returnDict["finalMask"])
            if "finalHand" in returnDict.keys():
                cv2.imshow('Final Hand', returnDict["finalHand"])
            cv2.imshow('Drawing', returnDict["drawing"])
        
        #Flag for skeletonization
        if flags["flagSkeleton"] == True:
            if "heirarchy" in returnDict.keys():
                returnDict = segment.get_skeleton(returnDict)
                cv2.imshow("Skeleton", returnDict["skeleton"])
 
        #Flag for saving the images in their respective directories
        if flagSave == True and img_num < num_imgs:
            for dir_name, dir_path in zip(dir_list, datapaths):
                if len(returnDict[dir_name].shape) == 3:
                    img = returnDict[dir_name][startCoord[1]:endCoord[1], startCoord[0]:endCoord[0], :]
                else:
                    img = returnDict[dir_name][startCoord[1]:endCoord[1], startCoord[0]:endCoord[0]]
                cv2.imwrite(dir_path + '/' + dir_name + str(img_num) + '.jpg', img)
            img_num += 1
            print(img_num)
        
        #Pop up message once all the images are generated
        if img_num == num_imgs:
            msg.alert('Dataset Successfully created at location ' + root_dir, 'Dataset Created!')
            key = ord('q')
            
        #If the 's' key is pressed, the flag for saving the dataset turns True
        if key == ord('s'):
            flagSave = True 
        
        #If the 'q' key is pressed, the program ends
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

