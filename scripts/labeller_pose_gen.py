import os,sys

TRAKING_REPO_ROOT = "/home/t/Documents/projects/openspace/tracking_new/"
sys.path.append(TRAKING_REPO_ROOT)
currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

from vidgear.gears import WriteGear
import cv2 
import glob 
import json 
import torch
import torchvision.transforms as transforms
from models_ds.detector.yolov5.utils.datasets import letterbox
from models_ds.detector.yolov5.utils.general import (non_max_suppression, scale_coords, xyxy2xywh)
from models_ds.shelfnet import models
from runners.PoseRunner import PoseEstSimple
from models_ds.shelfnet.config import cfg as shelfnet_cfg
import time 
import pickle

CTX = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
import numpy as np 


def save_run(path, frames,poses):
    name = path + ".avi"
    output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":"fps=20"}
    writer = WriteGear(output_filename = name, compression_mode=False, logging=False, **output_params)

    print(f"saving to {name}")
    while(len(frames)>0):
        frame = frames.pop(0)
        writer.write(frame)
            
    writer.close()

    # save poses to pickle
    with open(path + ".pkl", "wb") as f:
        pickle.dump(poses, f)

def draw_poses(keypts, ori_img, vis_thres = 0.1):
    """
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    """
    pairs = [[0, 1], [2, 4],[3, 5],[6, 8],[10, 8],[6, 5],[7, 9],[7, 5], [12, 14], [16, 14],[12, 6], [6, 11],[13, 15]]

    scores = keypts[:, 2]
    select = scores > vis_thres 
    if(len(scores[select]) == 0):
        print("not enough keypoints")
        return

    body_poolypts = []
    for i in range(len(pairs)):
        if(keypts[pairs[i][0]][2] <= vis_thres):
            continue
        body_poolypts.append(keypts[pairs[i][0]][0:2])
        body_poolypts.append(keypts[pairs[i][1]][0:2])
        
    body_polypts = np.int32(body_poolypts)

    if len(body_polypts)>0:
        # cv2.fillPoly(msk, [np.array(keypts[body_polypts_][:,0:2],np.int32)],color=[255,255,255])
        # print(body_polypts.sh)
        cv2.polylines(ori_img, pts =[body_polypts], color=(255,255,255),isClosed=False, thickness=5, lineType=-1)

def get_image_batch(img, img_size = 640):
    img_batch = []
    img_bgr = img
    # Padded resize
    image_whole = letterbox(img_bgr,new_shape=img_size)[0]
    # Convert
    image_whole = image_whole[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image_whole = np.ascontiguousarray(image_whole)
    # numpy to tensor
    image_whole = torch.from_numpy(image_whole).to(CTX)
    image_whole = image_whole/255.0  # 0 - 255 to 0.0 - 1.0
    if image_whole.ndimension() == 3:
        image_whole = image_whole.unsqueeze(0)
    img_batch.append(image_whole)
    
    img_batch  = torch.cat(img_batch,dim=0)
    return img_batch

def draw_dets(dets, img):
    if(not (dets is None or len(dets) == 0) ):
        for det in dets[:, :4]:
            x1,y1, x2, y2 = det
            cv2.rectangle(img,(int(x1), int(y1)),(int(x2),int(y2)),(0,255,0),3)

if(__name__ == "__main__"):
    
    out_folder = "/home/t/Documents/projects/openspace/OpenSpaceCamera/out_folder/"
    imgs= sorted(glob.glob(out_folder + "*.jpg"))
    vid_count = 0
    classes = {"1":"take", "2":"put"}

    pose_cfg = TRAKING_REPO_ROOT+'models_ds/shelfnet/coco/shelfnet/shelfnet50_256x192_adam_lr1e-3.yaml'
    pose_weight_path = TRAKING_REPO_ROOT+'data/checkpoints/shelfnet/ShelfNet50_256x192_MS_COCO_Keypoints.pth'
    shelfnet_cfg.merge_from_file(pose_cfg)    
    pose_estimator = PoseEstSimple(shelfnet_cfg, pose_weight_path)
    exit = False
    curr_index = 0
    curr_poses = []
    current_frames = []
    old_vid_no= None
    while(True):
        vid_no = int(imgs[curr_index].split("_c_")[0].split("n_")[-1])
        if(old_vid_no is None):
            old_vid_no = vid_no            
        elif(vid_no != old_vid_no):
            # save the poses
            print("saving poses for vid_no: ", vid_no, "".join(imgs[curr_index].split("_")[:-1]))            
            path = out_folder + "poses_" + str(vid_no)
            
            save_run()
            old_vid_no = vid_no
             
        img = cv2.imread(imgs[curr_index], -1)        
        current_frames.append(img.copy())

        record = False          
        # load labels from json 
        with open(imgs[curr_index].replace(".jpg",".json"), 'r') as f:
            label = json.load(f)
        dets = []
        for shape in label["shapes"]:
            dets.append(shape["points"])
        dets = np.array(dets)
        det = dets.reshape(-1,4)
        
        if(len(det) > 0):
            # print(det)
            poses = pose_estimator(img, det)
            curr_poses.append(poses)
            for i in range(len(poses)):
                draw_poses(poses[i], img)
            draw_dets(det, img)
        else:
            curr_poses.append([])

        imshow =cv2.resize(img, (1024, 768))
        
        # cv2.rectangle(imshow, (0,0), (120, 15), (0,0,0),-1)            
        # cv2.rectangle(imshow, (0,15), (120, 30), (0,0,0),-1)
        # cv2.putText(imshow,"class="+str(vid_class), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))
        # cv2.putText(imshow,"recording="+("on" if record else "off"), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))
        cv2.imshow("frame", imshow)

        ret = cv2.waitKey(0)
        
        if(ret == 27):
            cv2.destroyAllWindows()
            exit = True
            break             
        elif(ret == ord("a")):
            curr_index =max(curr_index- 2, 0)                
        print(f"curr_index={curr_index}")
        curr_index += 1
    
        if(exit):
            break
