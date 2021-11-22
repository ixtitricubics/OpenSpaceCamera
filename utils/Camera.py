import threading
import cv2
import time 
import os 
import math
import numpy as np
from vidgear.gears import WriteGear
import itertools
from utils.utils import get_rectangle_positions, read_yaml
import utils.utils as utils
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

    def save_run(self, allow_duplicates=False):
        """
            if dont allow duplicates the number of frames may become differet among cameras.
        """
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
                # if(frame.id == old_id):continue  # if you want to remove dublications
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
    def __init__(self, ips, count_window, show_width, show_height, calibrate=False, fuse=True) -> None:
        self.ips = ips 
        self.count_window = count_window
        self.show_width = int(show_width)
        self.show_height = int(show_height)
        self.calibrate = calibrate
        self.frames = None 
        self.fuse = fuse
        self.h_count = int(math.sqrt(count_window))
        self.w_count = self.h_count + math.ceil((count_window-(self.h_count*self.h_count))/count_window)
        while(self.h_count * self.w_count< count_window):
            self.h_count+=1
        
        self.vis_thread = threading.Thread(target=self.run)
        self.showed_frames_count = 0
        self.exit = False
        self.big_frame = np.zeros((self.show_height*self.h_count, self.show_width*self.w_count, 3), dtype=np.uint8)
        self.locker = threading.Lock()
    
    def create_windows(self ):
        if(len(self.ips) == 1):
            self.name = self.ips[0]
        else:
            self.name = "big_frame"
            
        cv2.namedWindow(self.name) 
        if(self.calibrate):       
            self.points = []
            self.load_calibration_points()
        if(self.fuse):
            self.points = [[-1,-1] for _ in range(len(self.ips))]
            self.point = None
            self.load_calib_info()            
        if(self.fuse or self.calibrate):            
            cv2.setMouseCallback(self.name, self.mouse_event)
        
    def update_frames(self, frames):
        self.frames = frames 
    def start(self):
        self.vis_thread.start()
    def stop(self):
        self.exit = True 
    def undo(self):
        if(len(self.points) > 0):
            _ = self.points.pop()
                
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if(self.calibrate):
                self.points.append([x, y])
        if(event == cv2.EVENT_MOUSEMOVE):
            self.point = [x, y]


    def save_calibration(self,):
        # import yaml
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None

        pixel_points = self.points#list(itertools.chain(*self.points))
        world_points = [[0,0],[0,0],[0,0],[0,0]]
        data = dict(
            pixel_points =pixel_points,
            world_points =world_points,
            img_shape= [self.show_width, self.show_height]
        )
        with open(f'configs/calibrations/{self.name}.yml', 'w') as outfile:
            yaml.dump(data, outfile)
    def load_calib_info(self):
        f_path = f'configs/calibrations/calib_info.yaml'
        data = read_yaml(f_path)
        self.calib_info = data
        # import pdb;pdb.set_trace()
        
    def load_calibration_points(self):
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None
        
        f_path = f'configs/calibrations/{self.name}.yml'
        if(os.path.exists(f_path)):
            with open(f_path, 'r') as f:
                try:
                    data = yaml.load(f)
                    print(data)
                    self.points = data["pixel_points"]
                    self.world_points = data["world_points"]
                except Exception as exc:
                    print(exc)
    def find_camera(self, pt):
        """
        returns the index of camera
        """
        x = int(pt[0] / self.show_width)
        y = int(pt[1] / self.show_height)
        index = y * self.w_count  + x
        # print("find_cam,x,y=", x, y)
        # print("index", index)
        
        return index
    def run(self):
        self.create_windows()
        while(not self.exit):
            if(self.frames is not None):
                # if mode is fuse and point is not none then find the camera which point is located 
                # and calculate all other camera points by conveting
                if(self.fuse and not self.point is None):
                    cam_index  = self.find_camera(self.point)                    
                    if(cam_index < len(self.ips)):
                        # print("current camera is located at", self.ips[cam_index])
                        other_cameras = set(self.ips) - set(self.ips[cam_index])
                        # print(self.calib_info)
                        # convert current point to world coordinates
                        pt_current = [(self.point[0]%self.show_width) /self.show_width,
                                     (self.point[1]%self.show_height) /self.show_height,
                                     1]
                        w_pt = utils.convert_point(pt_current, np.float32(self.calib_info[self.ips[cam_index]]), self.calib_info["img_shape"])
                        # print(w_pt)
                        # self.points[0] = self.point

                        # # now convert all the other camera points to img points
                        other_cameras = list(other_cameras)
                        for ind, ip in enumerate(other_cameras):
                            pt = utils.convert_point(w_pt, np.float32(self.calib_info[self.ips[cam_index]]), inv=True)                            
                            pt = [pt[0]/self.calib_info["img_shape"][0], pt[1]/self.calib_info["img_shape"][1]]
                            curr_shape = self.frames[-1].img.shape
                            index = self.find_camera(pt)
                            # change these
                            
                            # pt = [pt[0] * curr_shape[0], pt[0] * curr_shape[1]]
                            # self.points[index] = pt 

                            # cv2.circle( self.frames[index].img, pt, 10, (255,0,0), thickness=-1)
                            
                if(len(self.ips) == 1):
                    self.big_frame = cv2.resize(self.frames[0].img, (self.show_width,self.show_height))
                else:                    
                    for i in range(len(self.frames)):
                        st_x = i%self.w_count
                        st_y = i//self.w_count
                        self.big_frame[st_y*self.show_height:(st_y+1)*self.show_height,st_x*self.show_width:(st_x+1)*self.show_width,:] = cv2.resize(self.frames[i].img, (self.show_width,self.show_height))
                if(self.calibrate):
                    for i in range(len(self.points)):
                        cv2.circle(self.big_frame, self.points[i], 2, (255,0,0), thickness=-1)
                # elif(self.fuse):
                #     for i in range(len(self.points)):
                #         cv2.circle(self.big_frame, self.points[i], 10, (255,0,0), thickness=-1)
                    
                self.showed_frames_count +=1
                cv2.imshow(self.name, self.big_frame)
                ret = cv2.waitKey(1)
                if(self.calibrate):
                    if(ret == ord('z') or ret == ord('Z')):
                        self.undo()
                    elif(ret == ord('e')):
                        self.save_calibration()
            else:
                time.sleep(0.04)