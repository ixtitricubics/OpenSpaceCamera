
from fvcore.common.config import CfgNode


cfg = CfgNode()
cfg.CAMERA = CfgNode()
cfg.CAMERA.PASSWORD = "@12DFG56qwe851"
cfg.CAMERA.USERNAME = "admin"
cfg.CAMERA.PORT = 554
cfg.CAMERA.CALIBRATE= False 
cfg.CAMERA.FUSE= not cfg.CAMERA.CALIBRATE
cfg.CALIBRATION = CfgNode()
cfg.CALIBRATION.sides = [66, 106]

cfg.SAVE=False 
cfg.SAVE_WIDTH = 640
cfg.SAVE_HEIGHT = 480
