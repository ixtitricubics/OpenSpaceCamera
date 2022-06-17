import numpy as np
import cv2
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
sys.path.insert(0, os.getcwd())
from utils import calib_tools
import os 
import pickle 

class Calibrator:
    def __init__(self,ip, chessboardSize, 
                    chessboard_original_size, 
                    frameSize = (1920, 1080)):
        """
        Args:
            ip: ip of the camera example: 192.168.1.111
            chessboardSize: size of chessboard
            chessboard_original_size: size of chessboard in real life (mm but i didnt see the difference) 
        """
        self.ip = ip
        self.image_folder = os.path.join("data","images", ip)
        
        self.frameSize = frameSize
        self.chessboardSize = chessboardSize

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3000, 0.001)


        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

        self.objp = objp * chessboard_original_size/1000
        #print(objp)

        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.
        self.images = sorted(glob.glob(self.image_folder+'/*.jpg'))
        print(self.image_folder)
        print(self.images)
        self.selected_imgs = []
        self.test_img = None
    
    def find_corners(self):

        for img_path in self.images:

            img = cv2.imread(img_path, 1)
            print(img_path)
            img = cv2.resize(img, self.frameSize)
            if(self.test_img is None):
                self.test_img = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, cv2.CALIB_CB_ADAPTIVE_THRESH )
            try:
                corners = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), self.criteria)
            except:
                print("failed to find chessboard")
                continue

            # If found, add object points, image points (after refining them)
            if ret:
                while(True):
                    
                    imgt = np.uint8(img.copy())
                    
                    # Draw and display the corners
                    try:
                        cv2.drawChessboardCorners(imgt, self.chessboardSize, corners, ret)
                    except Exception as e:
                        print(corners)
                        print(self.chessboardSize)
                        print("error while drawing")
                        print(e)
                        
                    cv2.imshow('img left', imgt)
                    ret = cv2.waitKey(100)
                    if(ret == ord('s')):
                        self.selected_imgs.append(img_path)
                        self.objpoints.append(self.objp.copy())
                        self.imgpoints.append(corners.copy())
                        break 
                    elif(ret == ord("n")):
                        break
            if ret == 27:
                break
    
    def show_undistorted(self, img, camera_matrix, distortion_coefficients, newCameraMatrixR, roi):
        distorted_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        undistorted_frame = cv2.undistort(
            distorted_frame, camera_matrix, distortion_coefficients, None, newCameraMatrixR,
        )
        roi_x, roi_y, roi_w, roi_h = roi
        cropped_frame = undistorted_frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        cv2.imshow("distorted %s" % (distorted_frame.shape,), distorted_frame)
        cv2.imshow("undistorted %s" % (undistorted_frame.shape,), undistorted_frame)
        cv2.imshow("cropped %s" % (cropped_frame.shape,), cropped_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def get_projection(self, K, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvecs)
        R_inv = R.T
        T = tvecs
        H = np.eye(4)
        H[:3, :3] = R_inv
        H[:3, 3] = (R_inv@-T).T
        # Orientation vector
        O = np.matmul(R_inv, np.array([0, 0, 1]).T)
        return H, R, O
    def get_projection_error(self, objpoints, cameraMatrix, dist, rvecs, tvecs):
        # 49 x 1 x 2
        # 49 x 2 

        
        errors = []
        for idx in range(len(self.objpoints)):
            H,_,_ = self.get_projection(cameraMatrix, rvecs[idx], tvecs[idx])
            img_pts = calib_tools.project(self.objpoints[idx], cameraMatrix, H, dist)
            print(idx, (np.abs(self.imgpoints[idx].reshape(-1,2) - img_pts)).sum()/len(self.imgpoints[idx]), "\t", self.selected_imgs[idx])
            # print()
            errors.append((np.abs(self.imgpoints[idx].reshape(-1,2) - img_pts)).sum()/len(self.imgpoints[idx]))
        errors=np.array(errors)
        return errors 
    def calibrate(self):
        self.find_corners()
        print("--------------------------------------")
        print()
        print("finding corners finished", len(self.objpoints))
        print()
        print("--------------------------------------")
        
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.frameSize, None, None)
        errors = self.get_projection_error(self.objpoints, K, dist,rvecs, tvecs)
        print("initial errors")
        print(errors)
        max_threshold = 1
        max_idx = np.argmax(errors)

        while(errors[max_idx] > max_threshold):            
            self.objpoints.pop(max_idx)    
            self.imgpoints.pop(max_idx)
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.frameSize, None, None)
            errors = self.get_projection_error(self.objpoints, K, dist,rvecs, tvecs)
            max_idx = np.argmax(errors)
        print("after ransac errors")
        print(errors)
        width, height = self.frameSize
        optimalK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1, (width, height))
        self.show_undistorted(self.test_img, K, dist, optimalK, roi)
        
        data = {
            "optimalK":optimalK,
            "dist":dist,
            "K":K,
            "width":width,
            "height":height,
            }
        with open(f"data/intrinsics/{self.ip}.pickle", "wb") as f:
            pickle.dump(data, f)

if __name__  == "__main__":
    import sys
    ip = sys.argv[1]
    c = Calibrator(ip,chessboardSize = (8,6), chessboard_original_size=80)
    c.calibrate()