
from fvcore.common.config import CfgNode


cfg = CfgNode()
cfg.CAMERA = CfgNode()
cfg.CAMERA.PASSWORD = "@12DFG56qwe851"
cfg.CAMERA.USERNAME = "admin"
cfg.CAMERA.PORT = 554
cfg.CAMERA.CALIBRATE= True

cfg.SAVE=False
cfg.SAVE_WIDTH = 1024
cfg.SAVE_HEIGHT = 768
