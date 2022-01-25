import os 
import sys 
import cv2 


if(__name__ == '__main__'):
    
    print(sys.argv[1])
    cap = cv2.VideoCapture(sys.argv[1])
    shape = None 
    count = 0
    while(True):
        ret, frame = cap.read()
        if(ret):
            if(shape is None):
                shape = frame.shape
            count += 1
        else:
            break
        
    print("shape", shape)
    print("count", count)