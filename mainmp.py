import cv2
from utils.CameraMP import Camera, Visualization, Saver
import time 
import os 
from configs.camera import cfg as camera_config
import termios
import tty 
import sys 
import select 
from multiprocessing import Queue, SimpleQueue, JoinableQueue, Manager, Value
import multiprocessing
def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


if(__name__ == '__main__'):
    ips = ["192.168.1.101", "192.168.1.102","192.168.1.103","192.168.1.104"] # ,  
    cam_queues = {}
    save_queues = {}
    for ip in ips:
        cam_queues[ip] = Queue(maxsize=1)
        save_queues[ip] = Queue(maxsize=1)
    exit = Value('i', 0)

    cameras = [Camera(camera_config, cam_queues[ip],save_queues[ip], exit, ip) for ind, ip in enumerate(ips)]
    
    
    show_width = camera_config.SAVE_WIDTH
    show_height = camera_config.SAVE_HEIGHT
    vis = Visualization(ips, cam_queues, len(ips), show_width=show_width, show_height=show_height, 
                        calibrate=camera_config.CAMERA.CALIBRATE, fuse=camera_config.CAMERA.FUSE)
    saver = Saver(save_queues, ips, camera_config.SAVE_WIDTH, camera_config.SAVE_HEIGHT)

    # start reading
    for cam in cameras:
        cam.start()
    vis.start()
    saver.start()

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    count = 0
    save = False
    finish = False
    while(True):
        if(isData()):
            c = sys.stdin.read(1)
            if(c == 'e'):
                # for cam in cameras:
                #     cam.stop()     
                # vis.stop() 
                exit.value = 1
                break
            print("input::", c)
            
        time.sleep(0.1)
    print("main loop is ended")
    
    saver.wait()
    # for cam in cameras:
    #     cam.cam_process.join()
    # vis.vis_process.join()

    
    # if(camera_config.SAVE):
    #     for cam in cameras:
    #         cam.wait_for_saving()
