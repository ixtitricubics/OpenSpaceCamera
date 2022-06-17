import os 
import sys 
sys.path.insert(0,os.getcwd())
from utils.utils import  get_rectangle_positions, load_calibration,save_yaml, read_yaml
from utils.calib_tools import calibrate_homography
from configs.camera import cfg as camera_config

if(__name__ == '__main__'):
    ips = ["192.168.0.111","192.168.0.112","192.168.0.113","192.168.0.114","192.168.0.115"]
    H = read_yaml("data/calibrations/camera_info.yaml")
    for ip in ips:
        pix_pts, w_pts, img_shape, orig_shape= load_calibration(ip)
        print(ip)
        print(pix_pts, w_pts, img_shape, orig_shape)
        h = calibrate_homography(pix_pts, w_pts)
        H[ip]["H"] = h.tolist()
        H[ip]["img_shape"]= img_shape
        H[ip]["orig_shape"]= orig_shape        
        print(h)
    save_path = "configs/calibrations/camera_info.yaml"
    save_yaml(H, save_path)