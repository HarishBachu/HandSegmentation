import cv2
from SegmentationApp import segmentor

if __name__ == "__main__":
    detector = segmentor()
    detector.start_detection()