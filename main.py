import cv2
from utils.Camera import Camera, Visualization
from utils import utils
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
    ips = ["192.168.0.111"] #,"192.168.0.111","192.168.0.112","192.168.0.113","192.168.0.114","192.168.0.115"
    calib_info = utils.read_yaml("configs/calibrations/camera_info.yaml")
    cameras = {}
    for ip in ips:
        cameras[ip] = Camera(camera_config, calib_info[ip], ip)

    vis = Visualization(ips, len(ips),  camera_config.SHOW_WIDTH,
     camera_config.SHOW_HEIGHT, camera_config.CAMERA.CALIBRATE, 
     camera_config.CAMERA.FUSE, camera_config.CAMERA.SELECT_AREA, calib_info)
    # start reading 
    for ip in ips:
        cameras[ip].start()
    vis.start()
    # old_settings = termios.tcgetattr(sys.stdin)
    # tty.setcbreak(sys.stdin.fileno())

    # find the maximal delay 
    max_delay = 0
    for ip in ips:
        max_delay= max(calib_info[ip]["time_delay"],max_delay)
    
    count = 0
    save = False
    finish = False
    collected_frames = []
    delay = 1
    while(True):
        frames = {}
        start_read = time.time()

        for ip in ips:
            frame = cameras[ip]()
            if(frame is None or frame.img is None):
                break
            frames[ip] = frame
        
        
        time_read = round(time.time() - start_read,3)
        # print("frames", len(frames), frames.keys())
        if(len(frames) == len(cameras)):
            count +=1
            if(count >= delay):
                collected_frames.append(frames)
                if(len(collected_frames)>=max_delay):
                    new_frames = {}
                    for ip in ips:
                        new_frames[ip] = collected_frames[-calib_info[ip]["time_delay"]][ip]
                    collected_frames.pop(0)

                    start_save = time.time()
                    if(camera_config.SAVE):
                        for ip in ips:
                            cameras[ip].insert_frame_to_save(new_frames[ip])
                    time_save = round(time.time() - start_save,3)
                    vis.update_frames(new_frames)
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
            
        time.sleep(0.05)
    if(camera_config.SAVE):
        for ip in ips:
            cameras[ip].wait_for_saving()
