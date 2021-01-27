import numpy as np
import cv2
import imutils
import math

class segmentor:

    def __init__(self):

        #Flags
        self.flagMotion = False
        self.flagSkin = False 
        self.flagContour = False
        self.flagEllipse = False
        self.flagSkeleton = False 
        
        #Display Windows
        self.back = None
        self.drawing = None
        self.final_mask = None
        self.frame = None
        self.blur = None 
        self.motionMask = None
        self.motion = None 
        self.skinMask = None
        self.skin = None  
        self.hand = None 
        self.ellipseHand = None
        self.skeleton = None

        #Segmentation Hyperparameters
        self.BgSubtractor = None
        self.bgSubtractorLr = 0
        self.bgSubthreshold = 20
        self.minLim = np.array([0, 133, 77], dtype = np.uint8)
        self.maxLim = np.array([255, 173, 127], dtype = np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.epsilon = np.random.randn(1)/1000
        self.skelKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    #Built In Background Subtractor by Opencv
    #Learning Rate of 0 prevents white areas from being updated, detects motion
    def detect_motion(self):
        self.motionMask = self.bgSubtractor.apply(self.blur, learningRate = self.bgSubtractorLr)
        self.motionMask = cv2.morphologyEx(self.motionMask, cv2.MORPH_OPEN, self.kernel)
        self.motionMask = cv2.morphologyEx(self.motionMask, cv2.MORPH_CLOSE, self.kernel)
        self.motionMask = cv2.GaussianBlur(self.motionMask, (5, 5), cv2.BORDER_DEFAULT)
        _, self.motionMask = cv2.threshold(self.motionMask, 200, 255, cv2.THRESH_BINARY)
        self.motion = cv2.bitwise_and(self.frame, self.frame, mask = self.motionMask)

    # Skin Colors show distinct colors in YCrCb colorspace.
    # Pixels within a certain intensity range are captured. 
    # Processing the Motion mask for skin areas gives the moving skin parts. 
    def detect_skin(self):
        self.skinMask = cv2.cvtColor(self.motion, cv2.COLOR_BGR2YCR_CB)
        self.skinMask = cv2.GaussianBlur(self.skinMask, (5, 5), cv2.BORDER_DEFAULT)
        self.skinMask = cv2.inRange(self.skinMask, self.minLim, self.maxLim)
        self.skinMask = cv2.morphologyEx(self.skinMask, cv2.MORPH_OPEN, self.kernel)
        self.skinMask = cv2.morphologyEx(self.skinMask, cv2.MORPH_CLOSE, self.kernel)
        self.skinMask = cv2.GaussianBlur(self.skinMask, (5, 5), cv2.BORDER_DEFAULT)
        self.skin = cv2.bitwise_and(self.frame, self.frame, mask = self.skinMask)

    # Finds Center of the Contour 
    def get_moments(self, c):
        M = cv2.moments(c)
        cx = int(M["m10"]/(M["m00"] + self.epsilon))
        cy = int(M["m01"]/(M["m00"] + self.epsilon))
        return cx, cy

    # Defines motion of hand into 8 quadrants. Returns quadrant position of 
    # hand in current frame with respect to previous frame.
    def get_quadrant(self):
        px, py = self.prev
        nx, ny = self.now
        d = np.sqrt((px - nx)**2 + (py - ny)**2)
        
        if nx == px:
            theta = 90
        else:
            theta = np.degrees(np.arctan((ny - py)/(nx - px)))
            if theta < 0:
                theta = 180 + theta
        if d >= 5:
            if theta <= 45:
                if nx >= px and ny >= py:
                    return '8'
                elif nx <= px and ny <= py:
                    return '4'
            elif theta > 45 and theta <= 90:
                if nx >= px and ny >= py:
                    return '7'
                elif nx <= px and ny <= py:
                    return '3'
            if theta > 90 and theta <= 135:
                if nx <= px and ny >= py:
                    return '6'
                elif nx >= px and ny <= py:
                    return '2'
            if theta > 135 and theta <= 180:
                if nx <= px and ny >= py:
                    return '5'
                elif nx >= px and ny <= py:
                    return '1'
        else: 
            return None
    
    # Calculates angle between 2 points
    def calculateAngle(self, far, start, end):
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        return angle
    
    # Code to count fingers based on Contours and Convexity Defects
    def countFingers(self):
        contourAndHull = self.drawing 
        hull = cv2.convexHull(self.c)
        a_cnt = cv2.contourArea(self.c)
        a_hull = cv2.contourArea(hull)
        hull = cv2.convexHull(self.c, returnPoints = False)
        if (a_cnt*100/a_hull) > 90:
            return True, 0
        
        if len(hull):
            defects = cv2.convexityDefects(self.c, hull)
            cnt = 0
            if type(defects) != type(None):
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(self.c[s, 0])
                    end = tuple(self.c[e, 0])
                    far = tuple(self.c[f, 0])
                    angle = self.calculateAngle(far, start, end)

                    # Ignore the defects which are small and wide
                    # Probably not fingers
                    if d > 10000 and angle <= math.pi/2:
                        cnt += 1
                        cv2.circle(contourAndHull, far, 8, [255, 0, 0], -1)
                        cv2.imshow('ContourHull', contourAndHull)
            return True, cnt + 1
        return False, 0

    # Opencv code to find contours from skinMask. Returns the final mask.
    # Final mask is a representation of where the largest contour is. 
    # For Hand detection, the largest contour is almost always the hand. 

    def get_contour(self):
        contours, self.heirarchy = cv2.findContours(self.skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if self.heirarchy is not None:
            self.c = max(contours, key = cv2.contourArea)
            hull = cv2.convexHull(self.c)
            cx, cy = self.get_moments(self.c)
            self.now = (cx, cy)
            quadrant = self.get_quadrant()
            
            cv2.drawContours(self.final_mask, [self.c], 0, 255, -1)
            cv2.drawContours(self.drawing, [self.c], 0, (255, 255, 255), 2)
            cv2.drawContours(self.drawing, [hull], 0, (0, 0, 255), 2)
            cv2.circle(self.drawing, (cx, cy), 7, (255, 255, 255), -1)
            
            self.hand = cv2.bitwise_and(self.frame, self.frame, mask = self.final_mask)
            found, count = self.countFingers()
            
            cv2.line(self.hand, self.prev, self.now, (255, 0, 255), 2)
            cv2.putText(self.hand, str(count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(self.hand, 'Quadrant {}'.format(quadrant), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            self.prev = self.now

    
    # Calculates Best Fit ellipse around largest contour
    def best_fit(self):
        
        self.ellipseHand = self.hand
        extLeft = tuple(self.c[self.c[:,:,0].argmin()][0])
        extRight = tuple(self.c[self.c[:,:,0].argmax()][0])
        extTop = tuple(self.c[self.c[:,:,1].argmin()][0])
        extLeftBot = tuple(self.c[self.c[:,:,1].argmax()][0])
        diff = extTop[0] - extLeftBot[0]
        extRightBot = (extTop[0] + 2*diff, extLeftBot[1])

        cv2.drawContours(self.ellipseHand, [self.c], -1, (0, 255, 255), 2)
        cv2.circle(self.ellipseHand, extLeft, 8, (0, 0, 255), -1)
        cv2.circle(self.ellipseHand, extRight, 8, (0, 255, 0), -1)
        cv2.circle(self.ellipseHand, extTop, 8, (255, 0, 0), -1)
        cv2.circle(self.ellipseHand, extLeftBot, 8, (255, 255, 0), -1)
        cv2.circle(self.ellipseHand, extRightBot, 8, (100, 100, 100), -1)

        point_list = np.array([extLeft,extRight,extTop,extLeftBot,extRightBot])
        ellipse = cv2.fitEllipse(point_list)
        cv2.ellipse(self.ellipseHand, ellipse, (0, 0, 255), 2)

    #Finds skeleton of the final mask. Can be used for counting fingers 
    def get_skeleton(self):
        
        binary = self.final_mask
        self.skeleton = np.zeros(binary.shape, dtype = np.uint8)
        while True:
            open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.skelKernel)
            temp = cv2.subtract(binary, open)
            erode = cv2.erode(binary, self.skelKernel)
            self.skeleton = cv2.bitwise_or(self.skeleton, temp)
            binary = erode.copy()
            if cv2.countNonZero(binary) == 0:
                break


    #Starts Camera. Keypress 'b' starts segmentation algorithm. Keypress 'q' ends the program.
    def start_detection(self):

        self.cap = cv2.VideoCapture(0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.back = np.zeros((self.height, self.width, 3))
        self.prev = (0, 0)
    
        while self.cap.isOpened():
            self.now = (0, 0)
            self.drawing = np.zeros((self.height, self.width, 3), dtype = np.uint8)
            self.final_mask = np.zeros((self.height, self.width), dtype = np.uint8)
            self.frame = self.cap.read()[1]
            self.blur = cv2.GaussianBlur(self.frame, (5, 5), cv2.BORDER_DEFAULT)
            cv2.imshow('Frame', self.frame)
            cv2.imshow('Blurred', self.blur)

            if self.flagMotion == True:
                self.detect_motion()
                cv2.imshow('Motion Mask', self.motionMask)
                cv2.imshow('Moving Objects', self.motion) 
            
            if self.flagSkin == True:
                self.detect_skin()
                cv2.imshow('Skin Mask', self.skinMask)
                cv2.imshow('Moving Skin Parts', self.skin)
            
            if self.flagContour == True:
                self.get_contour()
                cv2.imshow('Final Mask', self.final_mask)
                if self.hand is not None:
                    cv2.imshow('Hand', self.hand)
                cv2.imshow('Drawing', self.drawing)
            
            if self.flagEllipse == True:
                if self.heirarchy is not None:
                    self.best_fit()
                    cv2.imshow('Ellipse Orientation', self.ellipseHand)

            if self.flagSkeleton == True: 
                if self.heirarchy is not None:
                    self.get_skeleton()
                    cv2.imshow('Skeleton', self.skeleton)

                    
            key = cv2.waitKey(1) & 0xFF

            if key == ord('b'):
                self.bgSubtractor = cv2.createBackgroundSubtractorMOG2(10, self.bgSubthreshold)
                self.flagMotion = True
                self.back = self.frame
                self.flagSkin = True 
                self.flagContour = True
                self.flagEllipse = True
                self.flagSkeleton = True 


            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    
    def get_frame(self):
        frame = self.frame 
        return frame
