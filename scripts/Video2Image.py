import cv2
import numpy as np
import os
import time

if __name__ == "__main__":
    import sys
    ip = sys.argv[1]
    vid_name = f"data/videos/{ip}.avi"
    print(vid_name)
    cap = cv2.VideoCapture(vid_name)    
    while(True):
        ret, frame = cap.read()

        cv2.imshow('img left', cv2.resize(frame, (1024, 768)))
        ret = cv2.waitKey(0)
        if(ret == ord('s')):
            name = str(time.time()) + ".jpg"
            cv2.imwrite(os.path.join("images",ip, name), frame)
            print(os.path.join("images",ip, name))
        if(ret == 27):
            break