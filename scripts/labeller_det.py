import os,sys 
TRAKING_REPO_ROOT = "/home/t/Documents/projects/openspace/tracking_new/"
sys.path.append(TRAKING_REPO_ROOT)
currentUrl = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(currentUrl, 'yolov5')))

from vidgear.gears import WriteGear
import cv2 
import glob 

import torch
import torchvision.transforms as transforms
from models_ds.detector.yolov5.utils.datasets import letterbox
from models_ds.detector.yolov5.utils.general import (non_max_suppression, scale_coords, xyxy2xywh)
from models_ds.shelfnet import models
from runners.PoseRunner import PoseEstSimple
from models_ds.shelfnet.config import cfg as shelfnet_cfg
import json
import time 
import pickle
import copy

CTX = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
import numpy as np 
sample_rect = {
      "label": "person",
      "points": [
        [
          300.88888888888886,
          97.22222222222221
        ],
        [
          1138.6666666666667,
          1070.5555555555554
        ]
      ],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
}
sample_label = {
  "version": "4.6.0",
  "flags": {},
  "shapes": [
  ],
  "imagePath": "n_0000_c_001_000.jpg",
  "imageData": None,
  "imageHeight": 1080,
  "imageWidth": 1440
}

def save_run(path, frames,dets):
    # name = path + ".avi"
    # output_params = {"-vcodec": "mpeg4", "-crf": "28", "-preset": "medium", "-filter:v":"fps=20"}
    # writer = WriteGear(output_filename = name, compression_mode=False, logging=False, **output_params)

    # print(f"saving to {name}")
    # while(len(frames)>0):
    #     frame = frames.pop(0)
    #     writer.write(cv2.resize(frame, (int(save_width), int(save_height))))
            
    # writer.close()
    
    
    for frame_id, frame in enumerate(frames):
        label = copy.deepcopy(sample_label)
        cv2.imwrite(f"{path}_{str(frame_id).zfill(3)}.jpg", frame)
        print(f"{path}_{str(frame_id).zfill(3)}.jpg")
        det = dets[frame_id]

        if(len(det)>0):
            for rect in det:
                x1,y1, x2, y2, threshold, class_no = rect 
                if(not class_no == 0):
                    raise Exception("class_no should be 0")
                rect= copy.deepcopy(sample_rect)
                rect["points"] = [[x1,y1],[x2,y2]]
                rect["flags"] = {"threshold": threshold}
                label["shapes"].append(rect)
        label["imagePath"] = f"{path}_{str(frame_id).zfill(3)}.jpg"
        label["imageHeight"] = frame.shape[0]
        label["imageWidth"] = frame.shape[1]
        with open(f"{path}_{str(frame_id).zfill(3)}.json", "w") as f:
            json.dump(label, f, indent=4)
        

    # save dets to pickle
    # with open(path + ".pkl", "wb") as f:
    #     pickle.dump(dets, f)

def _xywh_to_xyxy(bbox_xywh, width, height):
    x,y,w,h = bbox_xywh
    x1 = max(int(x-w/2),0)
    x2 = min(int(x+w/2),width-1)
    y1 = max(int(y-h/2),0)
    y2 = min(int(y+h/2),height-1)
    return x1,y1,x2,y2

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
        # cv2.imshow("ori_img", msk)
        # print("poses are drawn")

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
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


if(__name__ == "__main__"):
    video_folder = "saved/"
    videos= glob.glob(video_folder + "*.avi")
    out_folder = "/home/t/Documents/projects/openspace/OpenSpaceCamera/out_folder/"
    vid_count = 0
    last_name = sorted(glob.glob(out_folder + "*.jpg"))
    if(len(last_name) > 0):
        vid_count = int(last_name[-1].split("_c_")[0].split("n_")[-1])+1
    SAVE_WIDTH = 1920
    SAVE_HEIGHT = 1080
    classes = {"1":"take", "2":"put"}

    detector = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    detector.to(CTX).eval()

    pose_cfg = TRAKING_REPO_ROOT+'models_ds/shelfnet/coco/shelfnet/shelfnet50_256x192_adam_lr1e-3.yaml'
    pose_weight_path = TRAKING_REPO_ROOT+'data/checkpoints/shelfnet/ShelfNet50_256x192_MS_COCO_Keypoints.pth'
    shelfnet_cfg.merge_from_file(pose_cfg)    
    pose_estimator = PoseEstSimple(shelfnet_cfg, pose_weight_path)

    for video in videos:
        cap = cv2.VideoCapture()
        cap.open(video)
        frames = []
        curr_dets = []
        record = False          
        vid_class = -1
        curr_frame_id = 0
        while(True):
            ret, frame = cap.read()
            if(not ret or frame is None):
                break
            frame = image_resize(frame, height=SAVE_HEIGHT)
            frame_to_save = frame.copy()
            orig_shape = frame.shape
            curr_frame_id +=1
            
            if(record):
                frames.append(frame_to_save)

            batch = get_image_batch(frame.copy(),640)
            
            #run the detection
            with torch.no_grad():
                preds = detector(batch, augment=False)[0]  # list: bz * [ (#obj, 6)]
            preds = non_max_suppression(preds, 0.6, 0.5,
                                            classes=[0], agnostic=False, ips = ["sample"])  # for human only
            
            if(len(preds) > 0):
                det = preds["sample"]                    
                
                if(det is None or len(det) == 0):
                    continue 
                if len(det)>=1:
                    # Rescale boxes from img_size to original im0 size
                    confs = det[:, 4:5].cpu()
                    det[:, :4] = scale_coords(batch[0].cpu().numpy().shape[-2:], det[:, :4], orig_shape).round()
                    if(record):
                        curr_dets.append(det.cpu().numpy().tolist())
                    
                    if(True):                        
                        poses = pose_estimator(frame.copy(), det)
                        for i in range(len(poses)):
                            draw_poses(poses[i], frame)
                

                draw_dets(det, frame)
            elif(record):
                    curr_dets.append([])

            imshow = cv2.resize(frame, (1024, 768))
            cv2.rectangle(imshow, (0,0), (120, 15), (0,0,0),-1)
            
            cv2.rectangle(imshow, (0,15), (120, 30), (0,0,0),-1)
            cv2.putText(imshow,"class="+str(vid_class), (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))
            cv2.putText(imshow,"recording="+("on" if record else "off"), (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255))
            cv2.imshow("frame", imshow)
            ret = cv2.waitKey(0)

            if(chr(ret) == "s"):
                record = True 
                frames = []
                curr_dets = []
            elif(chr(ret) == "e"):
                # new path 
                name = "n_"+str(vid_count ).zfill(4) + "_c_" + str(vid_class).zfill(3)
                new_out_path = os.path.join(out_folder, name)
                if(len(frames) == len(curr_dets)):
                    save_run(new_out_path, frames, curr_dets)                    
                else:
                    print("frames and dets not same length")
                frames = []
                curr_dets = []                    
                print("saving finished", len(frames))
                record = False 
                vid_count +=1
            elif(chr(ret) in classes):
                vid_class = int(chr(ret))
                print("vid_class", vid_class)
            elif(ret == 27):
                cv2.destroyAllWindows()
                cap.release()
                break             
            elif(ret == ord("a")):
                curr_frame_id =max(curr_frame_id- 5, 0)                
                cap.set(1, curr_frame_id)
            print(f"curr_frame_id={curr_frame_id}")

