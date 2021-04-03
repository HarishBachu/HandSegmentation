import numpy as np 
import cv2
import imutils
import math

class segmentor:

    def __init__(self):

        #Constant variables and kernels used for Segmentation

        #minLim refers to the lower limit of skin pixel intensities in YCrCb colorspace
        self.minLim = np.array([0, 133, 77], dtype = np.uint8)      

        #maxLim refers to the upper limit of skin pixel intensities in YCrCb colorspace
        self.maxLim = np.array([255, 173, 127], dtype = np.uint8)   
        #Skin Pixels are the pixels whose intensities fall within the range between minLim and maxLim        
        
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) #Kernel defines a "neighbourhood" around the given pixel.

        self.epsilon = np.random.randn(1)/1000      #Small epsilon to prevent division by 0
        self.skelKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))        #Kernel defined for skeletonization process


    #Arguments are passed to each function using a dictionary returnDict.
    #Corresponding outputs are added to the returnDict dictionary, which is returned as output.
    #This ensures that all required images are kept in one place, which is easily accessible.

    #Calculates angle
    def calculateAngle(self, far, start, end):
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        return angle

    #Defines a "center of mass" for the contour
    def get_moments(self, returnDict):
        c = returnDict["c"]
        M = cv2.moments(c)
        cx = int(M["m10"]/(M["m00"] + self.epsilon))
        cy = int(M["m01"]/(M["m00"] + self.epsilon))
        return cx, cy

    #Detects Moving objects from webcam feed using Background Subtraction
    def detect_motion(self, bgSubtractor, bgSubtractorLr, returnDict):

        frame = returnDict["frame"]
        blur = returnDict["blur"]
        motionMask = bgSubtractor.apply(blur, learningRate = bgSubtractorLr)
        motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_OPEN, self.kernel)
        motionMask = cv2.morphologyEx(motionMask, cv2.MORPH_CLOSE, self.kernel)
        motionMask = cv2.GaussianBlur(motionMask, (5, 5), cv2.BORDER_DEFAULT)
        _, motionMask = cv2.threshold(motionMask, 200, 255, cv2.THRESH_BINARY)
        motion = cv2.bitwise_and(frame, frame, mask = motionMask)
        
        returnDict["motionMask"] = motionMask
        returnDict["motion"] = motion
        return returnDict

    #Detects Skin Pixels in the YCrCb colorspace
    def detect_skin(self, returnDict):
        
        frame = returnDict["frame"]
        motion = returnDict["motion"]
        skinMask = cv2.cvtColor(motion, cv2.COLOR_BGR2YCR_CB)
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), cv2.BORDER_DEFAULT)
        skinMask = cv2.inRange(skinMask, self.minLim, self.maxLim)
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, self.kernel)
        skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, self.kernel)
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), cv2.BORDER_DEFAULT)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        
        returnDict["skinMask"] = skinMask
        returnDict["skin"] = skin
        return returnDict

    #Counts the number of fingers from convexity defects information
    def countFingers(self, returnDict):

        c = returnDict["c"]
        hull = cv2.convexHull(c)
        a_cnt = cv2.contourArea(c)
        a_hull = cv2.contourArea(hull)
        hull = cv2.convexHull(c, returnPoints = False)
        if (a_cnt*100/a_hull) > 90:
            return True, 0

        if len(hull):
            defects = cv2.convexityDefects(c, hull)
            cnt = 0
            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(c[s, 0])
                    end = tuple(c[e, 0])
                    far = tuple(c[f, 0])
                    angle = self.calculateAngle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > 10000 and angle <= math.pi/2:
                        cnt += 1
                        
            return True, cnt + 1
        return False, 0
    
    #Returns the largest contour after motion and skin detection. In most cases, the largest contour is the Hand itself
    def get_contour(self, returnDict, boundingBox):

        frame = returnDict["frame"]
        skinMask = returnDict["skinMask"]
        drawing = returnDict["drawing"]
        finalMask = returnDict["finalMask"]
        contours, heirarchy = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if heirarchy is not None:
            c = max(contours, key = cv2.contourArea)
            returnDict["c"] = c
            hull = cv2.convexHull(c)
            cx, cy = self.get_moments(returnDict)

            cv2.drawContours(finalMask, [c], 0, 255, -1)
            cv2.drawContours(drawing, [c], 0, (255, 255, 255), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 2)
            cv2.circle(drawing, (cx, cy), 7, (255, 255, 255), -1)

            hand = cv2.bitwise_and(frame, frame, mask = finalMask)
            _, count = self.countFingers(returnDict)

            returnDict["count"] = count
            returnDict["finalHand"] = hand
            returnDict["finalMask"] = finalMask
            returnDict["heirarchy"] = heirarchy

            boundingBox["cx"] = cx
            boundingBox["cy"] = cy

        return returnDict, boundingBox
        
    #Returns the skeletonized hand
    def get_skeleton(self, returnDict):

        binary = returnDict["finalMask"]
        skeleton = np.zeros(binary.shape, dtype = np.uint8)
        while True:
            m_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.skelKernel)
            temp = cv2.subtract(binary, m_open)
            erode = cv2.erode(binary, self.skelKernel)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = erode.copy()
            if cv2.countNonZero(binary) == 0:
                break
        returnDict["skeleton"] = skeleton
        return returnDict

    
