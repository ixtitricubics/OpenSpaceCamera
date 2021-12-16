
from fvcore.common.config import CfgNode


cfg = CfgNode()
cfg.CAMERA = CfgNode()
cfg.CAMERA.PASSWORD = "@12DFG56qwe851"
cfg.CAMERA.USERNAME = "admin"
cfg.CAMERA.PORT = 554
cfg.CAMERA.CALIBRATE= False
cfg.CAMERA.FUSE= True
cfg.CAMERA.SELECT_AREA=False

cfg.CALIBRATION = CfgNode()
cfg.CALIBRATION.sides = [66, 106]

cfg.SAVE=True
cfg.SAVE_WIDTH = -1
cfg.SAVE_HEIGHT = -1
cfg.SHOW_WIDTH = 640
cfg.SHOW_HEIGHT = 480