import threading
import cv2
import time 
import os 
import math
import numpy as np
from vidgear.gears import WriteGear
from vidgear.gears import CamGear
import itertools
from utils.utils import get_rectangle_positions, read_yaml, save_yaml
import utils.utils as utils
import copy 
import pickle 
class Frame:
    def __init__(self, img, id):
        self.img = img
        self.id = id
        self.time = time.time()
        
class Camera:
    def __init__(self, cfg, cam_config, ip_address="192.168.1.104") -> None:
        self.profile_no = 2
        self.ip = ip_address
        self.cam_thread = threading.Thread(target=self.read)
        # self.cam_grabber_thread = threading.Thread(target=self.grab)
        self.user_name = cfg.CAMERA.USERNAME#"admin"
        self.passwd = cfg.CAMERA.PASSWORD#"@12DFG56qwe851"
        self.port = cfg.CAMERA.PORT#554
        self.path =cam_config["path"]
        self.fps = cam_config["fps"]
        # self.path = "rtsp://192.168.1.106:554/profile2/media.smp"
        print(self.path)
        self.time_delay=cam_config["time_delay"]
        # if you use opencv
        self.stream = CamGear( 
            source=self.path, 
            # stream_mode=True,
            # time_delay=cam_config["time_delay"],
            # logging=True
        )
        # self.cap = cv2.VideoCapture()                
        # self.cap.open(self.path)

        # self.retreive_cam_infos()
        # self.curr_frames = []
        self.curr_frame = None
        
        self.locker = threading.Lock()
        self.locker_save = threading.Lock()
        # self.cam_lock = threading.Lock()
        # self.frame_ready = False
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

    def save_run_old(self, allow_duplicates=True):
        """
            if dont allow duplicates the number of frames may become differet among cameras.
        """
        name = f"cam{self.ip}_" + time.strftime("%Y_%m_%d_%H_%M_%S") + ".avi"
        
        # output = cv2.VideoWriter(os.path.join("saved", name),
        #                 cv2.VideoWriter_fourcc('M','J','P','G'), 
        #                 25,
        #                 (int(self.save_height), int(self.save_width)))
        output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":f"fps={self.fps}"}

        writer = WriteGear(output_filename = os.path.join("saved", name), compression_mode=False, logging=False, **output_params)

        print(f"saving to {name}")
        
        old_id = -1
        while(not self.exit):
            while(len(self.frames) > 0):
                self.locker_save.acquire()
                frame = self.frames.pop(0)
                self.locker_save.release()
                if(frame is None):continue
                # print(f"cam{self.ip}", frame.id, old_id)
                if(not allow_duplicates):
                    if(frame.id == old_id):continue  # if you want to remove dublications
                old_id = frame.id 
                # output.write(cv2.resize(frame, (int(self.save_height), int(self.save_width))))
                
                if(self.save_height > 0):
                    writer.write(cv2.resize(frame.img, (int(self.save_height), int(self.save_width))))
                else:
                    writer.write(frame.img)
                self.saved_frames_count += 1            
            time.sleep(0.01)
        print(f"saving to {name} finished." )
        print(f"{self.ip} read frames = {self.read_frames_count}; saved frames = {self.saved_frames_count}")
        # output.release()
        writer.close()



    def save_run1(self, allow_duplicates=True):
        """
            if dont allow duplicates the number of frames may become differet among cameras.
        """
        name = f"cam{self.ip}_" + time.strftime("%Y_%m_%d_%H_%M_%S") 
        vid_name = name + ".avi"
        picker_name = name + ".pickle"
        output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":f"fps={self.fps}"}
        writer = WriteGear(output_filename = os.path.join("saved", vid_name), compression_mode=False, logging=False, **output_params)


        print(f"saving to {name}")
        data = []
        old_id = -1
        while(not self.exit):
            while(len(self.frames) > 0):
                self.locker_save.acquire()
                frame = self.frames.pop(0)
                self.locker_save.release()
                # if(frame is None):continue
                # cv2.imwrite(str(self.saved_frames_count) + ".jpg", frame.img)

                # print(f"cam{self.ip}", frame.id, old_id)
                if(not allow_duplicates):
                    if(frame.id == old_id):continue  # if you want to remove dublications
                old_id = frame.id 
                data.append({"id":frame.id, "time":frame.time})
                
                if(self.save_height > 0):
                    writer.write(cv2.resize(frame.img, (int(self.save_height), int(self.save_width))))
                else:
                    writer.write(frame.img)
                self.saved_frames_count += 1            
            time.sleep(0.01)
        print(f"saving to {name} finished." )
        print(f"{self.ip} read frames = {self.read_frames_count}; saved frames = {self.saved_frames_count}")
        # save the data to a picke
        with open(os.path.join("saved", picker_name), "wb") as f:
            pickle.dump(data, f)
        writer.close()


    def save_run(self, allow_duplicates=True):
        """
            if dont allow duplicates the number of frames may become differet among cameras.
        """
        from datetime import datetime
        folder_name = time.strftime("%Y_%m_%d_%H_%M_%S") 
        if(not os.path.exists(os.path.join("saved", folder_name))):
            os.makedirs(os.path.join("saved", self.ip, folder_name), exist_ok=True)

        print(f"saving to {folder_name}")
        data = []
        old_id = -1
        while(not self.exit):
            while(len(self.frames) > 0):
                self.locker_save.acquire()
                frame = self.frames.pop(0)
                self.locker_save.release()
                # if(frame is None):continue
                # cv2.imwrite(str(self.saved_frames_count) + ".jpg", frame.img)

                # print(f"cam{self.ip}", frame.id, old_id)
                if(not allow_duplicates):
                    if(frame.id == old_id):continue  # if you want to remove dublications
                old_id = frame.id 
                data.append({"id":frame.id, "time":frame.time})
                name = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3] + ".jpg"
                if(self.save_height > 0):
                    #writer.write(cv2.resize(frame.img, (int(self.save_height), int(self.save_width))))
                    cv2.imwrite(os.path.join("saved", self.ip,  folder_name, name), cv2.resize(frame.img, (int(self.save_height), int(self.save_width))))
                else:
                    #writer.write(frame.img)
                    cv2.imwrite(os.path.join("saved", self.ip,  folder_name, name), frame.img)
                self.saved_frames_count += 1            
            time.sleep(0.01)
        print(f"saving to {name} finished." )
        print(f"{self.ip} read frames = {self.read_frames_count}; saved frames = {self.saved_frames_count}")
        # save the data to a picke
        # with open(os.path.join("saved", picker_name), "wb") as f:
        #     pickle.dump(data, f)
       # writer.close()



    def insert_frame_to_save(self, frame):
        self.frames.append(frame)
        
    # def retreive_cam_infos(self):
    #     self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #     self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    #     print("cam_info", self.width, self.height, self.fps)

    def start(self):
        # self.cam_thread.daemon = True
        # self.cam_grabber_thread.daemon = True
        # self.cam_grabber_thread.start()
        self.cam_thread.start()
        
        if(self.save):
            self.save_thread.start()
    def stop(self):
        self.exit = True
    
    # def grab(self):
    #     while(not self.exit):
    #         self.cam_lock.acquire()
    #         self.frame_ready = self.stream.grab()
    #         self.cam_lock.release()
    #         time.sleep(0.01)
        
    
    def read(self):
        self.stream.start()
        print("Camera {} is reading".format(self.ip))
        while(not self.exit):
            frame = self.stream.read()                
            if(not frame is None):
                self.locker.acquire()
                self.curr_frame = Frame(frame, self.read_frames_count)
                # self.curr_frames.append(Frame(frame, self.read_frames_count))
                self.locker.release()
                self.read_frames_count +=1
            else:
                print("image is not read")
                time.sleep(0.01)
        self.stream.stop()
        print("Camera {} is stopped".format(self.ip))
    
    def __call__(self, ):
        ret_frame = None        
        while(True):
            if(self.curr_frame is not None):
                ret_frame= self.curr_frame
                # self.curr_frames = []
                break
            else:
                time.sleep(0.005)
        return ret_frame
class Visualization:
    def __init__(self, ips, count_window, show_width, show_height, calibrate, fuse, select_area, cam_config) -> None:
        self.ips = ips 
        self.count_window = count_window
        self.show_width = int(show_width)
        self.show_height = int(show_height)
        
        self.calibrate = calibrate
        self.select_area= select_area        
        self.fuse = fuse
        self.cam_config = cam_config
        # find the maximum delay 
        max_delay = 0
        for ip in self.ips:
            max_delay= max(self.cam_config[ip]["time_delay"],max_delay)
        self.max_delay = max_delay            
        self.h_count = int(math.sqrt(count_window))
        self.w_count = self.h_count + math.ceil((count_window-(self.h_count*self.h_count))/count_window)
        while(self.h_count * self.w_count< count_window):
            self.h_count+=1
        
        self.vis_thread = threading.Thread(target=self.run)
        self.locker = threading.Lock()

        self.showed_frames_count = 0
        self.exit = False
        self.calib_info = None
        # self.frames = None 
        self.curr_frames = []
        self.saved_frames_count = 0
        self.big_frame = np.ones((self.show_height*self.h_count, self.show_width*self.w_count, 3), dtype=np.uint8) * 255
        
        # load the calibration info file
        self.load_calib_info()
    
    def create_windows(self ):
        if(len(self.ips) == 1):
            self.name = self.ips[0]
        else:
            self.name = "big_frame"
        
        cv2.namedWindow(self.name) 
        if(self.calibrate):       
            self.points = []
            self.world_points = None
            self.load_calibration_points()
        if(self.fuse):
            self.points = [[-1,-1] for _ in range(len(self.ips))]
            self.point = None            
        if(self.select_area):
            self.selected_points = []
            self.active_pt = -1
            if(len(self.ips) == 1):
                self.selected_points = self.calib_info[self.ips[0]]["selected_points"]


        if(self.fuse or self.calibrate or self.select_area):
            cv2.setMouseCallback(self.name, self.mouse_event)

        
    def update_frames(self, frames):
        # self.frames = frames 
        self.locker.acquire()
        self.curr_frames.append(frames)
        self.locker.release()

        if(len(self.curr_frames)>1): # self.max_delay
            self.locker.acquire()
            del self.curr_frames[0]
            self.locker.release()

    def start(self):
        self.vis_thread.start()
    def stop(self):
        self.exit = True 
    def undo(self):
        if(self.calibrate):
            if(len(self.points) > 0):
                _ = self.points.pop()
        # if(self.select_area):
        #     if(len(self.selected_points)>0):
        #         _ = self.selected_points.pop()
                
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if(self.calibrate):
                self.points.append([x, y])

        elif(event == cv2.EVENT_LBUTTONDOWN):
            if(self.select_area):
                # find the closest point 
                # and mark it as moving point
                x_ = x/ self.show_width
                y_ = y/self.show_height
                dists = [math.hypot(x_ - pt[0], y_-pt[1]) for pt in self.selected_points]
                self.active_pt= np.argmin(dists)
                print(self.active_pt)
        elif(event == cv2.EVENT_LBUTTONUP):
            if(self.select_area):
                self.active_pt = -1
        elif(event == cv2.EVENT_MOUSEMOVE):
            self.point = [x, y]
            if(self.select_area):
                if(self.active_pt>=0):
                    # print("changing self.selected_points from", self.selected_points[self.active_pt])
                    self.selected_points[self.active_pt] = (round(x/self.show_width,3), round(y/self.show_height,3))
                    # print("changing self.selected_points to", self.selected_points[self.active_pt])
    def save_selected_area(self):
        """
        this assumes that there is only one camera given
        """
        self.calib_info[self.ips[0]]["selected_points"] = self.selected_points
        f_path = f'configs/calibrations/camera_info.yaml'
        save_yaml(self.calib_info, f_path)
        
    def save_calibration(self, orig_shape):
        # import yaml
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None

        pixel_points = self.points#list(itertools.chain(*self.points))
        if(self.world_points is None):
            world_points = [[0,0],[0,0],[0,0],[0,0]]
        else:
            world_points = self.world_points
        data = dict(
            pixel_points =pixel_points,
            world_points =world_points,
            img_shape= [self.show_width, self.show_height],
            orig_shape= (orig_shape[:2])[::-1]
        )
        with open(f'configs/calibrations/{self.name}.yml', 'w') as outfile:
            yaml.dump(data, outfile)
    def load_calib_info(self):
        f_path = f'configs/calibrations/camera_info.yaml'
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
        else:
            print(f_path, " does not exist")
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
    def save_curr_frames(self, frames):
        """
        save current frames to saved_frames folder
        """
        if(not os.path.exists(f"data/saved_frames")):
            os.makedirs(f"data/saved_frames",exist_ok=True)
        
        for ip in frames:
            if( frames[ip] is not None):
                cv2.imwrite(f"data/saved_frames/{str(self.saved_frames_count).zfill(4)}_{ip}.jpg", frames[ip].img)
        self.saved_frames_count +=1

    def run(self):
        self.create_windows()
        orig_shape = None
        while(not self.exit):
            while(len(self.curr_frames)>0):# self.max_delay-1
                # take the frames according to their delay
                # frames = {}
                # for ip in self.ips:
                #     frames[ip] = self.curr_frames[-self.cam_config[ip]["time_delay"]][ip]
                
                self.locker.acquire()
                frames = self.curr_frames.pop(0)
                self.locker.release()

                # print(frames.keys())
                # self.locker.acquire()
                # frames = self.curr_frames
                # self.locker.release()
                
                ########################################################################
                # if you want to show the whole image originally                       #
                # self.show_height, self.show_width = frames[self.ips[0]].img.shape[:2]#
                ########################################################################
                
                # if mode is fuse and point is not none then find the camera which point is located 
                # and calculate all other camera points by converting
                if(self.fuse and not self.point is None):
                    points = {}
                    cam_index  = self.find_camera(self.point)                    
                    if(cam_index < len(self.ips)):
                        other_cameras = set(self.ips) - set([self.ips[cam_index]])

                        # convert current point to world coordinates
                        pt_current = [(self.point[0]%self.show_width) /self.show_width,
                                     (self.point[1]%self.show_height) /self.show_height,
                                     1]
                        w_pt = utils.convert_point(pt_current, np.float32(self.calib_info[self.ips[cam_index]]["H"]), self.calib_info[self.ips[cam_index]]["img_shape"])
                        pt_new = utils.convert_point(w_pt,  np.float32(self.calib_info[self.ips[cam_index]]["H"]), self.calib_info[self.ips[cam_index]]["img_shape"], inv=True)
                        print(w_pt)
                        pt_new = [int(pt_new[0] *self.calib_info[self.ips[cam_index]]["orig_shape"][0]), int(pt_new[1] *self.calib_info[self.ips[cam_index]]["orig_shape"][1])]
                        points[self.ips[cam_index]] =  pt_new[:2]

                        # now convert all the other camera points to img points
                        other_cameras = list(other_cameras)
                        for ind, ip in enumerate(other_cameras):
                            pt = utils.convert_point(w_pt, self.calib_info[ip]["H"],  self.calib_info[ip]["img_shape"], inv=True)
                            pt = [int(pt[0] *self.calib_info[ip]["orig_shape"][0]), int(pt[1] *self.calib_info[ip]["orig_shape"][1])]
                            points[ip] = pt
                        for ip in frames:
                            if(frames[ip] is not None and frames[ip].img is not None):
                                cv2.circle(frames[ip].img,points[ip], 10, (255,0,0), thickness=-1)
                        # import pdb;pdb.set_trace()
                if(orig_shape is None and not frames[list(frames.keys())[0]] is None):
                    orig_shape = frames[list(frames.keys())[0]].img.shape
                if(len(self.ips) == 1):                      
                    if(frames[list(frames.keys())[0]] is None or frames[list(frames.keys())[0]].img is None):continue
                    self.big_frame = cv2.resize(frames[list(frames.keys())[0]].img, (self.show_width,self.show_height))
                    if(self.select_area):
                        pts = []
                        for i in range(len(self.selected_points)):
                            x, y = int(self.selected_points[i][0] * self.show_width), int(self.selected_points[i][1] * self.show_height)
                            pts.append((x,y))
                        # print(np.int32([pts]))
                        msk = np.ones_like(self.big_frame)*112

                        cv2.fillPoly(msk, np.int32([pts]),color=(0,355,0))
                        self.big_frame = cv2.addWeighted(self.big_frame,0.7, msk, 0.3, 0)
                        for i in range(len(self.selected_points)):
                            cv2.rectangle(self.big_frame,pts[i],[pts[i][0]+25,pts[i][1]-25] , (0,0,0),-1)
                            cv2.putText(self.big_frame,str(i),pts[i], cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                    
                else:                    
                    for i in range(len(frames)):
                        st_x = i%self.w_count
                        st_y = i//self.w_count
                        if(frames[self.ips[i]] is not None):
                            self.big_frame[st_y*self.show_height:(st_y+1)*self.show_height,st_x*self.show_width:(st_x+1)*self.show_width,:] = cv2.resize(frames[self.ips[i]].img, (self.show_width,self.show_height))
                

                if(self.calibrate):
                    for i in range(len(self.points)):
                        cv2.circle(self.big_frame, self.points[i], 2, (255,0,0), thickness=-1)
                    
                self.showed_frames_count +=1
                cv2.imshow(self.name, self.big_frame)
                ret = cv2.waitKey(100)
                if(ret == ord('z') or ret == ord('Z')):
                    if(self.calibrate or self.select_area):
                        self.undo()
                elif(ret == ord('e')):
                    if(self.calibrate):
                        self.save_calibration(orig_shape)
                    elif(self.select_area):
                        self.save_selected_area()
                elif(ret == ord('s')):
                    self.save_curr_frames(frames)
            time.sleep(0.01)