import cv2 
import numpy as np
import os 
def get_rectangle_positions(pt, cfg, a_top = True):
    """
    returns rectangle positions given first position. It assumes that rectangle is parallel to x axis.
    if a_top is true then first element of cfg.CALIBRATION.sides is on the top side.
    """
    if(a_top):
        pt1 = pt
        pt2 = [pt1[0] + cfg.CALIBRATION.sides[0], pt1[1]]
        pt3 = [pt1[0] + cfg.CALIBRATION.sides[0], pt1[1] + cfg.CALIBRATION.sides[1]]
        pt4 = [pt1[0], pt1[1] + cfg.CALIBRATION.sides[1]]
        return [pt1, pt2, pt3, pt4]
    else:
        pt1 = pt
        pt2 = [pt1[0] + cfg.CALIBRATION.sides[1], pt1[1]]
        pt3 = [pt1[0] + cfg.CALIBRATION.sides[1], pt1[1] + cfg.CALIBRATION.sides[0]]
        pt4 = [pt1[0], pt1[1] + cfg.CALIBRATION.sides[0]]
        return [pt1, pt2, pt3, pt4]

def calibrate(pix_pts, w_pts):
    h, status = cv2.findHomography(np.float32(pix_pts), np.float32(w_pts))
    return h

def convert_point(point_px, h, im_shape=None, inv=False):
    if(not im_shape is None):
        pt = [point_px[0] * im_shape[0], 
                point_px[1] * im_shape[1], 
                1]
    else:
        pt = [*point_px, 1]
    print("*** initial_pt", point_px, im_shape)
    print("*** converting point", pt)
    if(inv):
        pt = np.dot(np.linalg.inv(h), pt)
    else:
        pt = np.dot(h, pt)
        # pt = cv2.perspectiveTransform(np.float32([pt]), h)
    pt = (pt/pt[-1])[:2]
    res = np.int32(pt)
    print("****res", res)
    return res 

def save_yaml(dict_data, file_path):
    import ruamel.yaml
    yaml = ruamel.yaml.YAML()
    yaml.version = (1,2)
    yaml.default_flow_style = None

    with open(file_path, 'w') as outfile:
        yaml.dump(dict_data, outfile)

def read_yaml(f_path):
        import ruamel.yaml
        yaml = ruamel.yaml.YAML()
        yaml.version = (1,2)
        yaml.default_flow_style = None
        
        if(os.path.exists(f_path)):
            with open(f_path, 'r') as f:
                try:
                    data = yaml.load(f)
                    return data
                except Exception as exc:
                    print(exc)
        return None, None
        

def load_calibration(name):
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
                    img_shape= data["img_shape"]
                    return points, world_points, img_shape
                except Exception as exc:
                    print(exc)
        return None, None
        