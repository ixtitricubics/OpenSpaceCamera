import numpy as np 
import cv2 
import math 
def get_fundamental_matrix_from_prjs(ips, extrinsics):
        P_w_c1 = get_P(extrinsics[ips[0]]["R"], extrinsics[ips[0]]["T"])
        P_w_c2 = get_P(extrinsics[ips[1]]["R"], extrinsics[ips[1]]["T"])
        projection_matrix = np.array( [ [ 1, 0, 0, 0],
                                [ 0, 1, 0, 0],
                                [ 0, 0, 1, 0]] )
        try:
            F = cv2.sfm.fundamentalFromProjections(extrinsics[ips[0]]["K"]@ (projection_matrix @ P_w_c1) , extrinsics[ips[1]]["K"] @ (projection_matrix @ P_w_c2))
        except:
            F = fundamentalFromProjections(extrinsics[ips[0]]["K"]@ (projection_matrix @ P_w_c1), extrinsics[ips[1]]["K"] @ (projection_matrix @ P_w_c2))
        F = F/F[-1,-1]
        return F

def fundamentalFromProjections(P1, P2):
    X = []
    X.append(np.vstack([P1[1], P1[2]]))
    X.append(np.vstack([P1[2], P1[0]]))
    X.append(np.vstack([P1[0], P1[1]]))
    X = np.stack(X, axis=0)
    Y = []
    Y.append(np.vstack([P2[1], P2[2]]))
    Y.append(np.vstack([P2[2], P2[0]]))
    Y.append(np.vstack([P2[0], P2[1]]))
    Y = np.stack(Y, axis=0)
    F = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            XY = np.vstack([X[j], Y[i]])
            F[i, j] = cv2.determinant(XY)
    return F
    
def distortion(points_2d, K, dist_coeffs=None):
    if dist_coeffs is None:
        return points_2d

    k1, k2, p1, p2, k3 = dist_coeffs
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # To relative coordinates
    x = (points_2d[:, 0] - cx) / fx
    y = (points_2d[:, 1] - cy) / fy
    r2 = x * x + y * y

    # Radial distorsion
    xdistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    ydistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    xdistort += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    ydistort += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Back to absolute coordinates.
    xdistort = xdistort * fx + cx
    ydistort = ydistort * fy + cy

    return np.stack([xdistort, ydistort]).T

def project_simple(points_3d, K, H, dist_coeffs=None):
    """
    H is a projection matrix to convert from world to camera coordinate system.
    """
    def make_3x4(K, H):
        tmp = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
        return K @ tmp @ H
    P = make_3x4(K, H)
    p3d_ = np.hstack((points_3d, np.ones([len(points_3d), 1], dtype=points_3d.dtype)))
    p2d_ = p3d_ @ P.T
    p2d = p2d_[:, 0:2] / p2d_[:, 2:3]

    # only valid point needs distortion
    valid = np.all(p2d > 0, axis=1) & np.all(p2d < 1, axis=1)
    valid_p2d = p2d
    p2d = distortion(valid_p2d, K, dist_coeffs[0])

    return np.squeeze(p2d)


def project(points_3d, K, Tw, dist_coeffs=None):
    """
    Tw is a projection matrix to convert from camera to world coordinate system.
    """
    def make_3x4(K, Tw):
        tmp = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
        return K @ tmp @ np.linalg.inv(Tw)
    P = make_3x4(K, Tw)
    p3d_ = np.hstack((points_3d, np.ones([len(points_3d), 1], dtype=points_3d.dtype)))
    p2d_ = p3d_ @ P.T
    p2d = p2d_[:, 0:2] / p2d_[:, 2:3]

    # only valid point needs distortion
    valid = np.all(p2d > 0, axis=1) & np.all(p2d < 1, axis=1)
    valid_p2d = p2d
    p2d = distortion(valid_p2d, K, dist_coeffs[0])

    return np.squeeze(p2d)
def calibrate_homography(pix_pts, w_pts):
    h, status = cv2.findHomography(np.float32(pix_pts), np.float32(w_pts))
    return h
def normalize(K, w, h):
    """Normalizes the intrinsic matrix K by the given width and height"""
    return np.diag([1.0 / w, 1.0 / h, 1.0]) @ K
def undistort_img(image, K, dist):
    frame = cv2.undistort(image, K, dist, None)
    return frame
def normalize_pt(pt, K):
    """
    normalizes the point 
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    new_pt = [(pt[0]-cx)/fx, (pt[1]-cy)/fy]
    return new_pt
def get_P(R,T):
    P = np.eye(4)
    P[:3,:3] = R
    P[:3,3] = T.T
    return P
def get_cross_product_mtx(T1):
    """
    get the cross product matrix using a vector T1
    
    """
    mtx = np.zeros((3,3))
    
    # mtx[0, 0] = 0
    mtx[0, 1] = -T1[2]
    mtx[0, 2] = T1[1]
    
    mtx[1, 0] = T1[2]
    # mtx[1, 1] = 0
    mtx[1, 2] = -T1[0]
    
    mtx[2, 0] = -T1[1]
    mtx[2, 1] = T1[0]
    # mtx[2, 2] = 0
    return mtx 
def pseudo_inv(self, P):
    """
    receives rectangular 3x4 shaped matrix and calculates the inverse of it
    inverse matrix of rectangular matrix is totally different than doing inverse of rotation and translation one by one.
    """
    
    p_inv = np.linalg.pinv(P)
    return p_inv

def get_orientation_vect(H, O):
    arrow_length = H[:,-1][2] * 0.2
    xs = [H[:,-1][0], H[:,-1][0] - O[0] * arrow_length]
    ys = [H[:,-1][1], H[:,-1][1] - O[1] * arrow_length]
    zs = [H[:,-1][2], H[:,-1][2] - O[2] * arrow_length]
    return xs,ys,zs
# PNP 
def get_pose(cam_info):
    """
    cam_info: Dictionary
        keys: world_points, pixel_points, K, dist
    """
    ret,rvecs, tvecs = cv2.solvePnP(np.ascontiguousarray(cam_info["world_points"]).reshape((1,-1,3)), 
                                    np.ascontiguousarray(np.float32(cam_info["pixel_points"])).reshape((1,-1,2)),
                                    np.float32(cam_info["K"]), np.float32(cam_info["dist"]))
    R, _ = cv2.Rodrigues(rvecs)
    R_inv = R.T
    T = tvecs
    Hw2c = np.eye(4)
    Hw2c[:3, :3] = R
    Hw2c[:3, 3] = T.T

    Hc2w = np.eye(4)
    Hc2w[:3, :3] = R_inv
    Hc2w[:3, 3] = -((R_inv @ T).T)
    # Orientation vector
    O = np.matmul(R_inv, np.array([0, 0, 1]).T)
    return Hc2w,Hw2c, R,T, O
def distortion(points_2d, K, dist_coeffs=None):
    if dist_coeffs is None:
        return points_2d

    k1, k2, p1, p2, k3 = dist_coeffs
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    # # Back to absolute coordinates.
    # x = points_2d[:, 0]  * fx + cx
    # y = points_2d[:, 1]  * fy + cy

    # To relative coordinates
    x = (points_2d[:, 0]  - cx) / fx
    y = (points_2d[:, 1]  - cy) / fy
    r2 = x * x + y * y

    # Radial distorsion
    xdistort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    ydistort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    xdistort += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    ydistort += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # # Back to absolute coordinates.
    xdistort = xdistort * fx + cx
    ydistort = ydistort * fy + cy

    return np.stack([xdistort, ydistort]).T

def project(points_3d, K, Tw, dist_coeffs=None):
    def make_3x4(K, Tw):
        tmp = np.append(np.eye(3), np.zeros((3, 1)), axis=1)
        return K @ tmp @ np.linalg.inv(Tw)
    P = make_3x4(K, Tw)
    p3d_ = np.hstack((points_3d, np.ones([len(points_3d), 1], dtype=points_3d.dtype)))
    p2d_ = p3d_ @ P.T
    p2d = p2d_[:, 0:2] / p2d_[:, 2:3]

    valid_p2d = p2d
    p2d = distortion(valid_p2d, K, dist_coeffs[0])
    
    return np.squeeze(p2d)

def back_project(points_2d, z_worlds, K, Tw, dist_coeffs):
    """Back project points in the image plane to 3D
    A single point in the image plane correspods to a ray in 3D space. This
    method determines the 3D cooridates of the points where rays cast out
    of the image plane intersect with the provided heights.
    Args:
        points_2d (ndarray): An Nx2 array of image coordinates to back
                                project.
        z_worlds (ndarray): A list-like object of N heights (assuming z=0
                            is the ground plane) to back project to.
        K (ndarray): A 3x3 intrinsic matrix.
        Tw (ndarray): A 4x4 pose matrix.
        dist_coeffs (ndarray): An array of distortion coefficients of the form
                               [k1, k2, [p1, p2, [k3]]], where ki is the ith
                               radial distortion coefficient and pi is the ith
                               tangential distortion coeff.
    """
    # Unpack the intrinsics we are going to need for this calculation.
    fx, fy = K[0, 0], K[1, 1]
    ccx, ccy = K[0, 2], K[1, 2]
    points_2d = points_2d
    points_2d = cv2.undistortPoints(
        np.ascontiguousarray(np.float32(points_2d)).reshape((1,-1,2)), K, dist_coeffs, P=K
    ).squeeze(axis=1)
    points_3d = []
    # TODO: Vectorize
    for (x_image, y_image), z_world in zip(points_2d, z_worlds):
        kx = (x_image - ccx) / fx
        ky = (y_image - ccy) / fy
        # get point position in camera coordinates
        z3d = (z_world - Tw[2, 3]) / np.dot(Tw[2, :3], [kx, ky, 1])
        x3d = kx * z3d
        y3d = ky * z3d
        # transform the point to world coordinates
        x_world, y_world = (Tw @ [x3d, y3d, z3d, 1])[:2]
        points_3d.append((x_world, y_world, z_world))
    return np.array(points_3d)

def undistort(point_2d, K, dist):    
    point_2d = cv2.undistortPoints(
        np.ascontiguousarray(np.float32(point_2d)).reshape((1,-1,2)), K, dist, P=None
    ).squeeze(axis=1)[0]
    return point_2d

def get_projection_matrix(K, H):
    P = K @ np.eye(3, 4) @ H
    return P

def triangulate_with_optimization(points_2d, cam_ips, extrinsics, last_point_3d, threshold=0.4):
    err, new_pt = triangulate(points_2d, cam_ips, extrinsics)
    # last_point_3d = new_pt
    if(last_point_3d is not None):
        diff = np.linalg.norm(new_pt - last_point_3d)
        if(diff > threshold and len(points_2d) > 2):
            best_pt = None
            min_err_idx = -1
            for idx in range(len(points_2d)):
                points_2d_ = points_2d.copy()
                points_2d_.pop(idx)
                cam_ips_ = cam_ips.copy()
                cam_ips_.pop(idx)
                err, pt = triangulate(points_2d_, cam_ips_, extrinsics)
                diff_ = np.linalg.norm(pt - last_point_3d)
                if(min_err_idx <0 or diff_ < diff):
                    min_err_idx = idx
                    best_pt = pt
            if(diff > threshold):
                print("triangulate_with_optimization: diff > 0.4", diff)
                return diff, 0.6*last_point_3d + 0.4*new_pt
            else:
                return diff, best_pt
    return 0.0, new_pt

def triangulate(points_2d, cam_ips, extrinsics):
    """
    Triangulation on multiple points from different cameras.
    args:
        points_2d: N x 2 np.ndarray of 2D points,
                    the points should be normalized by the image width and height,
                    i.e. the inputed x, y should be in the range of [0, 1]
        cam_ips: camera ip for each point comes from
    """
    assert len(points_2d) >= 2, "triangulation requires at least two cameras"

    points_2d = np.asarray(points_2d)
    A = np.zeros([len(points_2d) * 2, 4], dtype=float)
    for i, point in enumerate(points_2d):
        ip = cam_ips[i]
        upoint = undistort(point, extrinsics[ip]["K"], extrinsics[ip]["dist"])
        P = get_projection_matrix(extrinsics[ip]["K"], extrinsics[ip]["HW2C"])
        A[2 * i, :] = upoint[0] * P[2] - P[0]
        A[2 * i + 1, :] = upoint[1] * P[2] - P[1]


    B = A.T@A
    u, s, vh = np.linalg.svd(B)
    error = s[-1]
    X = vh[len(s) - 1]
    point_3d = X[:3] / X[3]

    return error, point_3d

def drawlines(img,lines):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c, _= img.shape
    for r in lines:
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [-10000, -(r[2]+r[0]*-10000)/r[1] ])
        x1,y1 = map(int, [10000, -(r[2]+r[0]*10000)/r[1] ])
        img = cv2.line(img, (x0,y0), (x1,y1), color,2)
    return img

def update_euler_angles(new_angles):
    rx, ry, rz = map(lambda r: r * np.pi / 180, new_angles)

    sa = np.sin(rx)
    ca = np.cos(rx)
    sb = np.sin(ry)
    cb = np.cos(ry)
    sg = np.sin(rz)
    cg = np.cos(rz)

    r11 = cb * cg
    r12 = cg * sa * sb - ca * sg
    r13 = sa * sg + ca * cg * sb
    r21 = cb * sg
    r22 = sa * sb * sg + ca * cg
    r23 = ca * sb * sg - cg * sa
    r31 = -sb
    r32 = cb * sa
    r33 = ca * cb

    R = np.ascontiguousarray(np.asarray([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]))
    return R

def euler_angles1(HW2C):
    rx = np.arctan2(HW2C[2, 1], HW2C[2, 2])
    ry = np.arctan2(-HW2C[2, 0], np.sqrt(HW2C[2, 1] ** 2 + HW2C[2, 2] ** 2))
    rz = np.arctan2(HW2C[1, 0], HW2C[0, 0])

    rx, ry, rz = map(lambda r: r * 180 / np.pi, [rx, ry, rz])
    return np.array([rx, ry, rz]).contigeous()
    
def euler_angles(RW2C):
    sy = math.sqrt(RW2C[0,0] * RW2C[0,0] +  RW2C[1,0] * RW2C[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(RW2C[2,1] , RW2C[2,2])
        y = math.atan2(-RW2C[2,0], sy)
        z = math.atan2(RW2C[1,0], RW2C[0,0])
    else :
        x = math.atan2(-RW2C[1,2], RW2C[1,1])
        y = math.atan2(-RW2C[2,0], sy)
        z = 0
    rx, ry, rz = map(lambda r: r * 180 / np.pi, [x, y, z])
    return [rx, ry, rz]