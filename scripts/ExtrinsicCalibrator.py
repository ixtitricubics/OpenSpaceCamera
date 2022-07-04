
import cv2
import numpy as np
import os 
import sys 
sys.path.insert(0, os.getcwd())
import pickle 
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import math
from utils import calib_tools, calib_tools_plucker
from utils import utils as tools

import time

def get_image_pairs(ip1, ip2):
    image_pairs= {
        "192.168.0.111": "data/images_extrinsics/111.jpg",
        "192.168.0.112": "data/images_extrinsics/112.jpg",
        "192.168.0.113": "data/images_extrinsics/113.jpg",
        "192.168.0.114": "data/images_extrinsics/114.jpg",
        "192.168.0.115": "data/images_extrinsics/115.jpg",
        "192.168.0.116": "data/images_extrinsics/116.jpg",
        "192.168.0.117": "data/images_extrinsics/117.jpg",
    }
    return [image_pairs[ip1], image_pairs[ip2]]
# to get epipolar line we use these points 
x1,y1 = None, None
x2,y2 = None, None

class ExtrinsicCalibrator:
    def __init__(self, ips) -> None:
        self.load_info(ips)
        self.ips = ips
        self.test_fundamental_matrix=  True
        self.is_ok_save_results = False
        self.visualize_3d = False
        self.test_triangulate = False 

    def apply(self):
        # to get Rs and Ts
        self.get_poses(self.ips)
        # to test fundamental matrix
        if(self.test_fundamental_matrix):
            self.load_points_and_draw(self.ips[:2])
        
        # get all the fundamental matrices
        self.get_all_fundamental_matrices(self.ips)

        if(self.is_ok_save_results):
            self.save_results()
        
        # self.print_projection_error(self.ips)
        self.print_backprojection_results(self.ips)
        if(self.visualize_3d):
            self.visualize_results(self.ips)
        if(self.test_triangulate):
            self.triangulation_test()
        # self.print_projection_error(self.ips)

    def get_all_fundamental_matrices(self, ips):
        fundamental_matricess = {}
        for i in range(len(ips)-1):
            for j in range(i+1, len(ips)):
                F = calib_tools.get_fundamental_matrix_from_prjs((ips[i],ips[j]), self.cameras)
                fundamental_matricess[(ips[i], ips[j])] = F
                fundamental_matricess[(ips[j], ips[i])] = F.T                
        self.cameras["fundamental_matricess"] = fundamental_matricess
    
    def load_points_and_draw(self, ips):
        ips = (ips[0], ips[1])
        paths = get_image_pairs(ips[0], ips[1])
        img1 = cv2.imread(paths[0], -1)
        img2 = cv2.imread(paths[1], -1)

        print("starting to get the fundamental matrix")
        F1 = calib_tools.get_fundamental_matrix_from_prjs(ips, self.cameras)
        print('The new F1 = \n{}'.format(F1))
        print('det F1 = {}'.format(np.linalg.det(F1)))
        print("finished taking fundamental matrix")
        
        def mouse_event1(event, x, y, flags, param):        
            global x1,y1 
            if event == cv2.EVENT_LBUTTONDBLCLK:
                # TODO get the size of the image and resized image from configs
                x1,y1 = int(x * 1920/1024),int(y * 1080/768)
                print("x1,y1 changed", x1,y1)
        def mouse_event2(event, x, y, flags, param):
            global x2,y2 
            if event == cv2.EVENT_LBUTTONDBLCLK:
                x2,y2 = int(x * 1920/1024),int(y * 1080/768)
                print("x2,y2 changed", x2,y2)
        cv2.namedWindow("frame1") 
        cv2.setMouseCallback("frame1", mouse_event1)
        cv2.namedWindow("frame2") 
        cv2.setMouseCallback("frame2", mouse_event2)

        while(True):
            img1_copy = img1.copy()
            img2_copy = img2.copy()
            self.orig_shape = img1.shape

            if(x1 is not None):
                cv2.circle(img1_copy, (x1, y1), 10, (0, 0, 255), -1)
                undsp1 =  np.array([x1,y1])
                # pts2d_test,cam_centres= self.test_epipolar_lines_mocap(undsp1, ips[0], ips[1])
                lines1 = cv2.computeCorrespondEpilines(np.array(undsp1).reshape(-1, 1, 2), 1, F1)
                lines1 = lines1.reshape(-1, 3)
                img2_copy = calib_tools.drawlines(img2_copy, lines1)

            if(x2 is not None):
                cv2.circle(img2_copy, (x2, y2), 10, (0, 0, 255), -1)
                undsp2 =  np.array([x2,y2])
                lines2 = cv2.computeCorrespondEpilines(np.array(undsp2).reshape(-1, 1, 2), 1, F1.T)
                lines2 = lines2.reshape(-1, 3)
                img1_copy = calib_tools.drawlines(img1_copy, lines2)
            if(x1 is not None and x2 is not None):
                p1 = [*undsp1, 1.0]
                p2 = [*undsp2, 1.0]
                line1 = lines1[0]
                line2 = lines2[0]

                d1 = abs(np.array(p2) @ line1)/ math.sqrt(line1[0] * line1[0] + line1[1] * line1[1])/1000
                d2 = abs(np.array(p1) @ line2) / math.sqrt(line2[0] * line2[0] + line2[1] * line2[1])/1000
                print("d1 =", d1, "d2=",d2, "distanve = ", (d1+d2))
                self.test_epipolar_lines_mocap(p1, p2, ips[0], ips[1])
            
            cv2.imshow('frame1', cv2.resize(img1_copy, (1024,768)))
            cv2.imshow('frame2', cv2.resize(img2_copy, (1024,768)))

            ret = cv2.waitKey(10)
            if(ret == 27):
                break

        
    def test_epipolar_lines_mocap(self, keypoints1_2d, keypoints2_2d, cam_id1, cam_id2):
        
        cam1_invK = np.linalg.inv(self.cameras[cam_id1]["K"])
        cam1_R = self.cameras[cam_id1]["R"]
        cam1_T = self.cameras[cam_id1]["T"]
        
        cam2_invK = np.linalg.inv(self.cameras[cam_id2]["K"])
        cam2_R = self.cameras[cam_id2]["R"]
        cam2_T = self.cameras[cam_id2]["T"]

        keypoints1_2d = np.expand_dims(np.array([keypoints1_2d]), 0)
        keypoints2_2d = np.expand_dims(np.array([keypoints2_2d]), 0)

        lines1 = calib_tools_plucker.computeRay(keypoints1_2d,  cam1_invK, cam1_R, cam1_T)[0]
        lines2 = calib_tools_plucker.computeRay(keypoints2_2d,  cam2_invK, cam2_R, cam2_T)[0]

        
        lPluckers = []
        lPluckers.append(np.stack([lines1]))
        lPluckers.append(np.stack([lines2]))

        p0 = lPluckers[0][:, None]
        p1 = lPluckers[1][None, :]

        dist = calib_tools_plucker.dist_ll_pointwise_conf(p0, p1)

        print("distance", dist)

    def triangulation_test(self):
        ip = self.ips[np.random.randint(0, len(self.ips))]
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("-------------------STARTING TRIANGULATION FOR ", ip,'-------------------------')
        print("--------------------------------------------------------------------------------------")
        
        rest = list(set(self.ips) - set([ip]))
        print("AGAINST", rest)
        collected_points = []
        for point, pt_px in zip(self.cameras[ip]["world_points"], self.cameras[ip]["pixel_points"]):
            collected_points_ = [[point],[pt_px], [ip]]
            for ip_ in rest:
                # if(ip == ip_):continue
                for pt_3d, pt_2d in zip(self.cameras[ip_]["world_points"],self.cameras[ip_]["pixel_points"]):
                    if(pt_3d[0] == point[0] and pt_3d[1] == point[1] and pt_3d[2] == point[2]):
                        collected_points_[0].append(pt_3d)
                        collected_points_[1].append(pt_2d)
                        collected_points_[2].append(ip_)
            collected_points.append(collected_points_)
            # print("finished one point----------------")
        # print(collected_points)
        for collected_point in collected_points:
            if(len(collected_point[0]) > 1):
                print("starting to triangulate with", len(collected_point[1]), "points")
                err, pt_pr = calib_tools.triangulate(collected_point[1], collected_point[2], self.cameras)
                print("-------------------------------")
                print("error", err, collected_point[2])
                print("predicted_pt:::", pt_pr)
                print("correct pt:::", collected_point[0][0])
                # print("diff", pt_pr - collected_point[0][0])
                print("divv", collected_point[0][0]/pt_pr)
                print("-------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        print("--------------------------------------------------------------------------------------")
        

    def save_results(self):
        name = "extrinsics_.pickle"
        with open(os.path.join("data","extrinsics", name), "wb") as f:
            pickle.dump(self.cameras, f)
            print("saved to", os.path.join("data","extrinsics", name))

    def load_info(self, ips, use_old_extrinsics=True):
        """
        it assumes that there is a folder and pickle object that has the same name with ip        
        """
        
        cameras =  tools.read_yaml(os.path.join("data","calibrations","camera_info.yaml"))
        # print(dir(self.cameras))
        # print(type(self.cameras))
        if(use_old_extrinsics):
            with open(os.path.join("data", "extrinsics","extrinsics.pickle"), 'rb') as f:
                extrinsics = pickle.load(f)
        self.cameras = {}
        for ip in ips:                
            self.cameras[ip] = {}
            # it should have two arrays: pixel_points, world_points [sm]
            self.cameras[ip]["pixel_points"] = np.float32(cameras[ip]["pixel_points"])
            self.cameras[ip]["world_points"] = np.float32(cameras[ip]["world_points"])
            self.cameras[ip]["selected_area"] = cameras[ip]["selected_area"]
            # zeros = np.zeros((len(self.cameras[ip]["world_points"]), 1)) 
            # self.cameras[ip]["world_points"] = np.hstack((self.cameras[ip]["world_points"], zeros))
            
            if(not use_old_extrinsics):
                # load calibration
                with open(os.path.join("data", "intrinsics",ip +".pickle"), 'rb') as f:
                    intrinsics = pickle.load(f)
                self.cameras[ip]["K"] = intrinsics["K"] if "K" in intrinsics else intrinsics["cameraMatrix"]
                self.cameras[ip]["optimalK"] =  intrinsics["optimalK"] if "optimalK" in intrinsics else intrinsics["newCameraMatrix"]
                self.cameras[ip]["dist"] = intrinsics["dist"]
            else:
                self.cameras[ip]["K"] = np.float32(extrinsics[ip]["K"] if "K" in extrinsics[ip] else extrinsics[ip]["cameraMatrix"])
                self.cameras[ip]["optimalK"] =  np.float32(extrinsics[ip]["optimalK"] if "optimalK" in extrinsics[ip] else extrinsics[ip]["newCameraMatrix"])
                self.cameras[ip]["dist"] = extrinsics[ip]["dist"]

    def get_poses(self, ips):
        for ip in ips:
            Hc2w, Hw2c, R, T, O = calib_tools.get_pose(self.cameras[ip])
            self.cameras[ip]["HW2C"] = Hw2c
            self.cameras[ip]["HC2W"] = Hc2w
            self.cameras[ip]["R"] = R
            self.cameras[ip]["Orientation"] = O
            self.cameras[ip]["T"] = T
            print(ip, Hc2w[:,-1])

    def print_projection_error(self, ips):
        """
        ips: list of ip string
        """
        for ip in ips:
            print("-----------------------------------------------")
            print()
            print("printing results for camera", ip)
            new_pts = calib_tools.project(self.cameras[ip]["world_points"], 
                                self.cameras[ip]["K"], 
                                self.cameras[ip]["HC2W"],
                                self.cameras[ip]["dist"])
            print("projected points")
            print(np.int32(new_pts))
            print("original points")
            print(self.cameras[ip]["pixel_points"])
            print("projected - original")
            print(new_pts  - self.cameras[ip]["pixel_points"])
            print("reprojection_error", np.abs((new_pts  - self.cameras[ip]["pixel_points"])).sum()/len(new_pts))
            print("-----------------------------------------------")
            # print("ip", self.cameras[ip]["dist"])

    def print_backprojection_results(self, ips):
        print()
        # for ip in ips:
        ip = self.ips[0]
        z1_worlds = 0.0 * (self.cameras[ip]["world_points"][:, -1])/100

        new_pt = calib_tools.back_project(self.cameras[ip]["pixel_points"], z1_worlds, 
                    self.cameras[ip]["K"], 
                    self.cameras[ip]["HC2W"], 
                    self.cameras[ip]["dist"])
        print("backprojected points:::")
        print(new_pt)
        print("original points:::")
        print((self.cameras[ip]["world_points"])/100)

        new_pt = calib_tools.backProjectParallel(self.cameras[ip]["pixel_points"], z1_worlds, 
                    self.cameras[ip]["K"], 
                    self.cameras[ip]["HC2W"],
                    self.cameras[ip]["dist"])
        print("new methodd ")
        print(new_pt)            

    
    def visualize_results(self, ips):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d(-1, 5)
        ax.set_ylim3d(-2, 3)
        ax.set_zlim3d(-5, 0)

        for ip in ips:
            pts_3d = self.cameras[ip]["world_points"]
            ax.scatter(pts_3d[:,0]/100,pts_3d[:,1]/100,pts_3d[:,1]/100 * 0, c='blue')
            ax.scatter(*(self.cameras[ip]["HC2W"][:,-1])[:-1], marker="x", c="black")
            xs,ys,zs = calib_tools.get_orientation_vect(self.cameras[ip]["HC2W"], self.cameras[ip]["Orientation"])
            arrow1 = tools.Arrow3D(xs, ys, zs, mutation_scale=5, lw=2, arrowstyle="-|>", color="k")
            ax.add_artist(arrow1)
        plt.gca().set_aspect('auto', adjustable='box')
        plt.show()

if __name__ == "__main__":
    ips = [ 
            # "192.168.0.111",
            # "192.168.0.112",
            # "192.168.0.113",
            "192.168.0.114",
            # "192.168.0.115",
            "192.168.0.116",
            # "192.168.0.117"
        ]
    exC = ExtrinsicCalibrator(ips)
    exC.apply()