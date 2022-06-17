import numpy as np 
import cv2 
def estimate_line_using_plucker_method(point1, point2):
    """
    i have no fng ide what it does ::::::::::::::::)
    point1 is the camera center
    point2 is the all the other points 
    Returns:
        line_unit : 3 dim line vector
        moment: 3 dim line (a,b,c of ax+by+cz=0)
    """
    line = point2-point1 # i know that it is calculating line vector .
    norm = np.linalg.norm(line, axis=-1, keepdims=True) # this one calculates length  of line vector
    # convert the line vector to unit vetor
    line_unit = line/ norm
    # if we take any point in a line and take cross product it with direction vectro we should get a line (a,b,c) 
    moment = np.cross(point1, line_unit, axis=-1)
    return line_unit, moment

def computeRay(keypoints2d, invK, R, T):
        # points: (nJoints, 3)
        # invK: (3, 3)
        # R: (3, 3)
        # T: (3, 1)
        # cam_center: (3, 1)
        if len(keypoints2d.shape) == 3:
            keypoints2d = keypoints2d[0]
        conf = keypoints2d[..., -1:] # it takes the last column and makes column vector 
        cam_center = - R.T @ T # this one takes inverse of rotation and translation
        N = keypoints2d.shape[0]
        kp_pixel = np.hstack([keypoints2d[..., :2], np.ones_like(conf)])
        kp_all_3d = (kp_pixel @ invK.T - T.T) @ R
        l, m =  (cam_center.T, kp_all_3d)
        res = np.hstack((l, m, conf))
        return res[None, :, :]
def epipolar_lines_with_plucker_method(keypoints2d, invK, R, T):
    """
    i mostly did not understand what the fuck is happening tin the code but i am going to try anyway.
    keypoints2d : n x3        # we expect it to be undistorted
    """
    y = keypoints2d[..., -1:]
    cam_center = -R.T @ T # it just a inverse ot translation vector which points to camera
    kp_pixel = np.hstack([keypoints2d[..., :2], np.ones_like(y)]) # homogeneous vector x,y,1
    kp_all_3d = (kp_pixel @ invK.T - T.T) @ R # normalizing (returning back to world coordinate system) it gives 3d point located on the epipolar line in 3d world which should be common to all cameras
    l, m = estimate_line_using_plucker_method(cam_center.T, kp_all_3d) # this one then calculates line equation in 3d 
    res = np.hstack((l, m))
    return cam_center, m#res[None, :, :]
def dist_ll_pointwise(p0, p1):
    product = np.sum(p0[..., :3] * p1[..., 3:6], axis=-1) + np.sum(p1[..., :3] * p0[..., 3:6], axis=-1)
    return np.abs(product)

def dist_ll_pointwise_conf(p0, p1):
    dist = dist_ll_pointwise(p0, p1)
    conf = np.sqrt(p0[..., -1] * p1[..., -1])
    dist = np.sum(dist*conf, axis=-1)/(1e-5 + conf.sum(axis=-1))
    # dist[conf.sum(axis=-1)<0.1] = 1e5
    return dist