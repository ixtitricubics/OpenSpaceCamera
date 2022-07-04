import cv2
import numpy as np
import os
import sys 
import argparse
import time
import pickle
sys.path.insert(0, os.getcwd())
import utils.utils as tools
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_read_pts", action="store_true")
    parser.add_argument("--ip", type=str, default="192.168.0.111")
    parser.add_argument("--type", type=str, default="selected_area", choices=["selected_area", "pixel_points"])
    parser.add_argument("--img_path", type=str, default="data/images_extrinsics/111.jpg")
    return parser.parse_args()

args_sys = parse_args()

points = []
info =None 
def load_info():
    """
    types: str:
        can be selected_area, pixel_points
    """
    global points 
    global info 
    # load points
    info = tools.read_yaml(os.path.join("data","calibrations","camera_info.yaml"))
    # it should have two arrays: pixel_points, world_points [sm], selected_area
    points = info[args_sys.ip][args_sys.type]
    if(args_sys.scale_read_pts):
        for pt in points:
            pt[0] *= info[args_sys.ip]["orig_shape"][0]/info[args_sys.ip]["img_shape"][0]
            pt[1] *= info[args_sys.ip]["orig_shape"][1]/info[args_sys.ip]["img_shape"][1]
def save():
    assert (args_sys.type == "selected_area" and len(points) == 2) or (args_sys.type == "pixel_points" and len(points) >=4), f"error in selecting correct number of points type:{args_sys.type},num_pts:{len(points)}"
    info[args_sys.ip][args_sys.type] = points
    tools.save_yaml(info, os.path.join("data","calibrations","camera_info.yaml"))

def undo(points):
    if(len(points) > 0):
        _ = points.pop()

def mouse_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        new_x =  int(x* info[args_sys.ip]["orig_shape"][0]/info[args_sys.ip]["img_shape"][0])
        new_y =  int(y* info[args_sys.ip]["orig_shape"][1]/info[args_sys.ip]["img_shape"][1])
        if(args_sys.type == "selected_area" and len(points) >= 2):
            points.pop()
        # elif(args_sys.type == "pixel_points" and len(points) >= 4):
        #     points.pop()
        points.append([new_x, new_y])
        print("new point is added", len(points))


if __name__ == "__main__":
    print("IP::", args_sys.ip)
    print("img_path::", args_sys.img_path)
    print("scale_read_pts", args_sys.scale_read_pts)
    print("type:::", args_sys.type)
    img1 = cv2.imread(args_sys.img_path, -1)
    cv2.namedWindow(args_sys.type)
    cv2.setMouseCallback(args_sys.type, mouse_event)
    camera = load_info()
    while(True):
        img1_copy = img1.copy()
        pts1 = np.int32(points)
        # draw all the points
        for i in range(len(pts1)):            
            cv2.circle(img1_copy, (pts1[i][0], pts1[i][1]), 3, (0, 0, 255), -1)
            # draw the number 
            cv2.putText(img1_copy, str(i+1), (pts1[i][0], pts1[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # print(pts1[i])
        if(args_sys.type == "selected_area" and len(pts1) == 2):
            cv2.rectangle(img1_copy, pts1[0], pts1[1], (255,255,0), thickness=2)
        img1_copy = cv2.resize(img1_copy, info[args_sys.ip]["img_shape"])
        cv2.imshow(args_sys.type, img1_copy)
        ret = cv2.waitKey(10)
        if(ret == 27):
            break
        elif(ret == ord('z') or ret == ord('Z')):
            undo(points)
        elif(ret == ord('s') or ret == ord('S')):
            save()
