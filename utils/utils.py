import cv2 
import numpy as np
import os 
import sys
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



def print_error(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    print(e)

def convert_point(point_px, h, im_shape=None, inv=False):
    if(not im_shape is None and not inv):
        pt = [point_px[0] * im_shape[0], 
                point_px[1] * im_shape[1], 
                1]
    else:
        pt = [*point_px, 1]
    # print("*** initial_pt", point_px, im_shape)
    # print("*** converting point", pt)
    if(inv):
        pt = np.dot(np.linalg.inv(h), pt)
    else:
        pt = np.dot(h, pt)
        # pt = cv2.perspectiveTransform(np.float32([pt]), h)
    pt = (pt/pt[-1])[:2]
    res = np.int32(pt)
    # print("****res", res)
    if(inv and not im_shape is None):
        res = [res[0]/im_shape[0], res[1]/im_shape[1]]
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
        return None

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
                    orig_shape= data["orig_shape"]
                    return points, world_points, img_shape, orig_shape
                except Exception as exc:
                    print(exc)
        return None, None, None, None
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
class Arrow3D(FancyArrowPatch):
    '''
    3D arrow class can be shown in matplotlib 3D model.
    '''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    
    
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]),(xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
