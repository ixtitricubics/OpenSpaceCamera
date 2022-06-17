from fvcore.common.config import CfgNode


cfg = CfgNode()
cfg.CAMERA = CfgNode()
cfg.CAMERA.PASSWORD = "@12DFG56qwe851"
cfg.CAMERA.USERNAME = "admin"
cfg.CAMERA.PORT = 554
cfg.CAMERA.FUSE= False
cfg.CAMERA.CALIBRATE= False
cfg.CAMERA.SELECT_AREA=True

cfg.CALIBRATION = CfgNode()
cfg.CALIBRATION.sides = [66, 106]

cfg.SAVE=True
cfg.SAVE_FRAMES_ONCLICK=True
cfg.SAVE_WIDTH = -1
cfg.SAVE_HEIGHT = -1
cfg.SHOW_WIDTH = 1024
cfg.SHOW_HEIGHT = 768
