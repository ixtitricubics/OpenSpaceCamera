import threading
import cv2
import time 
import os 
import math
import numpy as np
from vidgear.gears import WriteGear

class Frame:
    def __init__(self, img, id):
        self.img = img
        self.id = id
        
class Camera:
    def __init__(self, cfg, ip_address="192.168.1.104") -> None:
        self.profile_no = 2
        self.ip = ip_address
        self.cam_thread = threading.Thread(target=self.read)
        self.user_name = cfg.CAMERA.USERNAME#"admin"
        self.passwd = cfg.CAMERA.PASSWORD#"@12DFG56qwe851"
        self.port = cfg.CAMERA.PORT#554
        self.path = f"rtsp://{self.user_name}:{self.passwd}@{self.ip}:{self.port}/profile{self.profile_no}/media.smp"
        # self.path = "rtsp://192.168.1.106:554/profile2/media.smp"
        print(self.path)
        self.cap = cv2.VideoCapture()                
        self.cap.open(self.path)
        self.retreive_cam_infos()
        # self.curr_frames = []
        self.curr_frame = None
        
        self.locker = threading.Lock()
        self.locker_save = threading.Lock()
        self.exit = False
        self.save = cfg.SAVE
        if(cfg.SAVE):
            self.frames = []
            self.save_thread = threading.Thread(target=self.save_run)
            self.saved_frames_count = 0
            self.save_width = cfg.SAVE_HEIGHT
            self.save_height = cfg.SAVE_WIDTH
        self.read_frames_count = 0

    def wait_for_saving(self):
        self.save_thread.join()
    def save_run(self):
        name = f"cam{self.ip}_" + time.strftime("%Y_%m_%d_%H_%M_%S") + ".avi"
        # output = cv2.VideoWriter(os.path.join("saved", name),
        #                 cv2.VideoWriter_fourcc('M','J','P','G'), 
        #                 25,
        #                 (int(self.save_height), int(self.save_width)))
        output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":"fps=20"}

        writer = WriteGear(output_filename = os.path.join("saved", name), compression_mode=False, logging=False, **output_params)

        print(f"saving to {name}")
        old_id = -1
        while(not self.exit or len(self.frames)>0):
            if(len(self.frames) > 0):
                self.locker_save.acquire()
                frame = self.frames.pop(0)
                self.locker_save.release()
                print(f"cam{self.ip}", frame.id, old_id)
                if(frame.id == old_id):continue 
                old_id = frame.id 
                # output.write(cv2.resize(frame, (int(self.save_height), int(self.save_width))))
                writer.write(cv2.resize(frame.img, (int(self.save_height), int(self.save_width))))
                self.saved_frames_count += 1
                
            else:
                time.sleep(0.05)
        print(f"saving to {name} finished." )
        print(f"{self.ip} read frames = {self.read_frames_count}; saved frames = {self.saved_frames_count}")
        # output.release()
        writer.close()

    def insert_frame_to_save(self, frame):
        self.frames.append(frame)
        
    def retreive_cam_infos(self):
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def start(self):
        self.cam_thread.start()
        if(self.save):
            self.save_thread.start()
    def stop(self):
        self.exit = True
    def read(self):
        print("Camera {} is reading".format(self.ip))
        while(not self.exit):
            self.cap.grab()
            self.locker.acquire()
            ret, frame = self.cap.retrieve()
            self.curr_frame = Frame(frame, self.read_frames_count)
            # self.curr_frames.append(curr_frame)
            
            self.locker.release()
            if not ret:
                print("image is not read")
                time.sleep(0.05)
            else:
                self.read_frames_count +=1
        self.cap.release()
        print("Camera {} is stopped".format(self.ip))
    
    def __call__(self, ):
        ret_frame = None        
        
        if(self.curr_frame is not None):
            self.locker.acquire()
            ret_frame= self.curr_frame
            # self.curr_frames = []
            self.locker.release()     
        return ret_frame

class Visualization:
    def __init__(self, count_window, show_width, show_height) -> None:
        self.count_window = count_window
        self.show_width = int(show_width)
        self.show_height = int(show_height)
        self.frames = None 
        self.h_count = int(math.sqrt(count_window))
        self.w_count = self.h_count + math.ceil((count_window-(self.h_count*self.h_count))/count_window)
        while(self.h_count * self.w_count< count_window):
            self.h_count+=1

        self.vis_thread = threading.Thread(target=self.run)
        self.showed_frames_count = 0
        self.exit = False
        self.big_frame = np.zeros((self.show_height*self.h_count, self.show_width*self.w_count, 3), dtype=np.uint8)
    def update_frames(self, frames):
        self.frames = frames 
    def start(self):
        self.vis_thread.start()
    def stop(self):
        self.exit = True 
    def run(self):
        while(not self.exit):
            if(self.frames is not None):
                for i in range(len(self.frames)):
                    st_x = i%self.w_count
                    st_y = i//self.w_count
                    self.big_frame[st_y*self.show_height:(st_y+1)*self.show_height,st_x*self.show_width:(st_x+1)*self.show_width,:] = cv2.resize(self.frames[i].img, (self.show_width,self.show_height))

                self.showed_frames_count +=1
                cv2.imshow("big_frame", self.big_frame)
                cv2.waitKey(1)
            else:
                time.sleep(0.04)