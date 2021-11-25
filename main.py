import cv2
from utils.Camera import Camera, Visualization
import time 
import os 
from configs.camera import cfg as camera_config
import termios
import tty 
import sys 
import select 

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


if(__name__ == '__main__'):
    ips = ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104"] #, "192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104"
    cameras = {}
    for ip in ips:
        cameras[ip] = Camera(camera_config, ip)
    
    show_width = camera_config.SAVE_WIDTH
    show_height = camera_config.SAVE_HEIGHT
    vis = Visualization(ips, len(ips), show_width, show_height, camera_config.CAMERA.CALIBRATE, camera_config.CAMERA.FUSE)
    # start reading
    for ip in ips:
        cameras[ip].start()
    vis.start()

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    count = 0
    save = False
    finish = False
    while(True):
        frames = {}
        start_read = time.time()
        for ip in ips:
            frame = cameras[ip]()
            if(frame is None):
                break
            frames[ip] = frame
        
        time_read = round(time.time() - start_read,3)

        if(len(frames) == len(cameras)):
            
            start_save = time.time()
            if(camera_config.SAVE):
                for ip in ips:
                    cameras[ip].insert_frame_to_save(frames[ip])
            time_save = round(time.time() - start_save,3)

            vis.update_frames(frames)

            spent_time = round(time.time() - start_read,3)
            # print(f'reading = {time_read} s. saving = {time_save}s. spent time {spent_time}')
        if(isData()):
            c = sys.stdin.read(1)
            if(c == 'e'):
                for ip in ips:
                    cameras[ip].stop()     
                vis.stop() 
                break              
            elif(c == 'z'):
                if(camera_config.CAMERA.CALIBRATE):
                    vis.undo()
            print("input::", c)
            
        time.sleep(0.025)
    if(camera_config.SAVE):
        for ip in ips:
            cameras[ip].wait_for_saving()
