from multiprocessing import queues
from multiprocessing.queues import Queue
import threading
import cv2
import time 
import os, sys
import math
import numpy as np
from numpy.linalg.linalg import inv
from vidgear.gears import WriteGear
import itertools
from utils.utils import get_rectangle_positions, read_yaml
import utils.utils as utils
from multiprocessing import Process, freeze_support
from functools import partial
class Frame:
    def __init__(self, img, id, ip, exit):
        self.img = img
        self.id = id
        self.ip = ip 
        self.exit = exit

class Saver:
    def __init__(self, in_queues,ips, save_width, save_height):
        self.in_queues = in_queues
        self.save_process = Process(target=self.run, args=(in_queues,ips, save_width, save_height))
    def start(self):
        self.save_process.start()
    
    @staticmethod
    def get_writer(ip, time_id):
        name = f"cam{ip}_" +time_id+ ".avi"
        output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":"fps=20"}

        writer = WriteGear(output_filename = os.path.join("saved", name), compression_mode=False, logging=False, **output_params)
        return writer

    def run(self, in_queues, ips, save_width, save_height):
        try:
            time_id =  time.strftime("%Y_%m_%d_%H_%M_%S") 
            writers = {}
            for ip in ips:
                writers[ip] = self.get_writer(ip, time_id)
            saved_frames = 0
            while(True):
                exit = False
                frames = {}
                for ip in ips:
                    try:
                        frame = in_queues[ip].get(block=True)
                    except:
                        continue
                    frames[ip] = frame
                    exit = exit or frame.exit
                if(len(frames) == len(ips)):
                    for ip in ips:
                        writers[ip].write(cv2.resize(frames[ip].img, (int(save_width), int(save_height))))
                    saved_frames += 1
                if(exit):
                    break
            for ip in ips:
                writers[ip].close()
            print(f"Saved {saved_frames} frames")
            for ip in ips:
                print(f"{ip} size = {in_queues[ip].qsize()}, {in_queues[ip].full()}")
                while(in_queues[ip].qsize() > 0):
                    _ = in_queues[ip].get()
        except Exception as e:
            utils.print_error(e)
        

    
    def wait(self):
        self.save_process.join()

class Camera:
    def __init__(self, cfg, out_queue,save_queue, exit, ip_address="192.168.1.104") -> None:
        self.profile_no = 2
        self.ip = ip_address
        self.user_name = cfg.CAMERA.USERNAME#"admin"
        self.passwd = cfg.CAMERA.PASSWORD#"@12DFG56qwe851"
        self.port = cfg.CAMERA.PORT#554
        self.path = f"rtsp://{self.user_name}:{self.passwd}@{self.ip}:{self.port}/profile{self.profile_no}/media.smp"
        # self.path = "rtsp://192.168.1.106:554/profile2/media.smp"
        print(self.path)
        
        self.cam_process = Process(target=self.read, args=[out_queue,save_queue, self.path, self.ip, exit])

    @staticmethod
    def retreive_cam_infos(cap):
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height =cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return [width, height, fps]

    def start(self):
        self.cam_process.start()
    def stop(self):
        self.cam_process.terminate()
    
    @staticmethod
    def read(out_queue,save_queue,  path, ip, exit):
        cap = cv2.VideoCapture() 
        try:
            cap.open(path)
            read_frames_count = 0
            print("Camera {} is reading".format(ip))
            while(True): # 
                ret, frame = cap.read()
                curr_frame = Frame(frame, read_frames_count, ip, exit.value)
                
                # self.curr_frames.append(curr_frame)
                if(out_queue.empty()):
                    out_queue.put(curr_frame)
                if(save_queue.empty()):
                    save_queue.put(curr_frame)
                if not ret:
                    print("image is not read")
                else:
                    read_frames_count +=1
                if(exit.value):
                    break 
            cap.release()
            print("Camera {} is stopped, read images count = {}".format(ip, read_frames_count))
        except Exception as e:
            utils.print_error(e)
            cap.release()
        print(f"{ip} out_queue size = {out_queue.qsize()}, {out_queue.full()}")
        print(f"{ip} save_queue size = {save_queue.qsize()}, {save_queue.full()}")
        # if(save_queue.full()):
        #     save_queue.get()
   
class Visualization:
    def __init__(self, ips, in_queues, count_window, show_width, show_height, calibrate=False, fuse=True) -> None:
        self.ips = ips 
        self.count_window = count_window
        self.show_width = int(show_width)
        self.show_height = int(show_height)
        self.calibrate = calibrate
        self.fuse = fuse
        self.h_count = int(math.sqrt(count_window))
        self.w_count = self.h_count + math.ceil((count_window-(self.h_count*self.h_count))/count_window)
        while(self.h_count * self.w_count< count_window):
            self.h_count+=1
        
        self.vis_process = Process(target=Visualization.run, args=[ in_queues,
                                                        [self.show_width, self.show_height], 
                                                           self.w_count, self.h_count, 
                                                           self.ips, 
                                                           calibrate,
                                                           fuse
                                                           ])
        self.showed_frames_count = 0
        self.exit = False
        
        self.locker = threading.Lock()
    
    @staticmethod
    def create_windows(ips, calibrate,fuse):
        if(len(ips) == 1):
            name = ips[0]
        else:
            name = "big_frame"
        
        cv2.namedWindow(name) 
        if(calibrate):       
            points, world_points = Visualization.load_calibration_points(name)
        else:
            points, world_points = None, None
        if(fuse):
            calib_info = Visualization.load_calib_info()
        else:
            calib_info = None
        return name, calib_info, points, world_points
        
    def start(self):
        self.vis_process.start()

    def stop(self):
        self.vis_process.terminate()

    @staticmethod
    def save_calibration(name, points_px, show_shape):
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None

        pixel_points = points_px
        world_points = [[0,0],[0,0],[0,0],[0,0]]
        data = dict(
            pixel_points =pixel_points,
            world_points =world_points,
            img_shape= show_shape
        )
        with open(f'configs/calibrations/{name}.yml', 'w') as outfile:
            yaml.dump(data, outfile)
    @staticmethod
    def load_calib_info():
        f_path = f'configs/calibrations/camera_info.yaml'
        data = read_yaml(f_path)
        calib_info = data
        return calib_info
        # import pdb;pdb.set_trace()
    @staticmethod
    def load_calibration_points(name):
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None
        
        f_path = f'configs/calibrations/{name}.yml'
        if(os.path.exists(f_path)):
            with open(f_path, 'r') as f:
                try:
                    data = yaml.load(f)
                    print(data)
                    points = data["pixel_points"]
                    world_points = data["world_points"]                    
                    return points, world_points
                except Exception as exc:
                    utils.print_error(exc)
        else:
            print(f"{f_path} does not exist")
        return None, None
        
    @staticmethod
    def find_camera(pt, show_shape, w_count):
        """
            returns the index of camera
        """
        x = int(pt[0] / show_shape[0])
        y = int(pt[1] / show_shape[1])
        index = y * w_count  + x
        return index
    @staticmethod
    def run(in_queues, show_shape, w_count, h_count, ips, calibrate, fuse):
        """
        Arguments:
            in_queues: list of queues # must be equal to len(ips)
            exit: Value object, 1 if the process is ended
            show_shape: shape of the frame to show
            w_count: number of windows in width
            h_count: number of windows in height
            ips: list of ip addresses
            calibrate: if true, turn on calibration mode
            fuse: if true, fuse the frames
        """
        try:
            show_width, show_height = show_shape                        
            points = {}            
            calib_info = None
            
            #returning calibration info
            name, calib_info, px_points, world_points = Visualization.create_windows(ips, calibrate, fuse)
            
            print(px_points)
            
            # global variables for for mouse event 
            point_dict = {}
            point_dict["points_px"] =px_points

            def mouse_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDBLCLK:
                    if(calibrate):
                        if(len(point_dict["points_px"]) == 4):
                            _ = point_dict["points_px"].pop()
                        point_dict["points_px"].append([x, y])
                if(event == cv2.EVENT_MOUSEMOVE):
                    point_dict["pt"]=[x, y]
                    # print("mouse is pressed")
            if(fuse or calibrate):            
                cv2.setMouseCallback(name,mouse_event)
            big_frame = np.zeros((show_height*h_count, show_width*w_count, 3), dtype=np.uint8)
            
            # to count the visualized frames, for debuggings
            showed_frames_count = 0
            while(True):
                frames = {}
                count_nones = 0
                exit = False
                for i in range(len(ips)):
                    try:
                        frame  = in_queues[ips[i]].get(timeout=1.0)
                        frames[ips[i]] = frame
                        exit = exit or frame.exit
                        if(not frame.ip == ips[i]):
                            print(frame.ip, ips[i])
                            print("ips didnt match")
                            import pdb;pdb.set_trace()
                    except Exception as e:
                        # utils.print_error(e)
                        pass 
                if(exit):
                    break 
                if(len(frames) == len(in_queues)):     
                    if("pt" in point_dict):
                        point = point_dict["pt"]
                    else:
                        point = None                 
                    # if mode is fuse and point is not none then find the camera which point is located 
                    # and calculate all other camera points by converting
                    if(fuse and not point is None):
                        curr_shape = frames[ips[0]].img.shape
                        cam_index  = Visualization.find_camera(point, show_shape, w_count)      
                        if(cam_index < len(ips)):
                            other_cameras = set(ips) - set([ips[cam_index]])
                            other_indexes = set(list(range(len(ips)))) - set([cam_index])
                            # convert current point to world coordinates
                            pt_current = [(point[0]%show_width) /show_width,
                                        (point[1]%show_height) /show_height,
                                        1]
                            w_pt = utils.convert_point(pt_current, np.float32(calib_info[ips[cam_index]]), calib_info["img_shape"])
                            pt_new = utils.convert_point(w_pt, np.float32(calib_info[ips[cam_index]]), calib_info["img_shape"], inv=True)
                            # pt_new_ = [int(pt_current[0] * curr_shape[1]), int(pt_current[1] * curr_shape[0])]
                            pt_new = [int(pt_new[0] *curr_shape[1]), int(pt_new[1] *curr_shape[0])]
                            points[ips[cam_index]] =  pt_new[:2]
                            # now convert all the other camera points to img points
                            other_cameras = list(other_cameras)
                            other_indexes = list(other_indexes)
                            for ind, ip in enumerate(other_cameras):
                                pt = utils.convert_point(w_pt, np.float32(calib_info[ip]),  calib_info["img_shape"], inv=True)
                                pt = [int(pt[0] *curr_shape[1]), int(pt[1] *curr_shape[0])]
                                points[ip] = pt
                            for i in range(len(ips)):
                                cv2.circle(frames[ips[i]].img, points[ips[i]], 10, (255,0,0), thickness=-1)
                    if(len(ips) == 1):
                        big_frame = cv2.resize(frames[name].img, (show_width,show_height))
                    else:                    
                        for i in range(len(ips)):
                            st_x = i%w_count
                            st_y = i//w_count
                            big_frame[st_y*show_height:(st_y+1)*show_height,st_x*show_width:(st_x+1)*show_width,:] = cv2.resize(frames[ips[i]].img, (show_width,show_height))
                    if(calibrate):
                        for i in range(len(px_points)):
                            cv2.circle(big_frame, px_points[i], 3, (255,0,255), thickness=-1)

                    showed_frames_count +=1
                    cv2.imshow(name, big_frame)
                    ret = cv2.waitKey(1)
                    if(calibrate):
                        if(ret == ord('z') or ret == ord('Z')):
                            if(len(px_points) > 0):
                                _ = px_points.pop()
                        elif(ret == ord('e')):
                            Visualization.save_calibration(name, px_points, show_shape)            
        except Exception as e:
            utils.print_error(e)
        print(f"visualization is ended, showed frames = {showed_frames_count}")
        for ip in ips:
            print(f"{ip} size = {in_queues[ip].qsize()}, {in_queues[ip].full()}")
            while(in_queues[ip].qsize() > 0):
                _ = in_queues[ip].get()