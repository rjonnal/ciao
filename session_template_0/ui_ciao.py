import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

if ccfg.simulate:
    sim = ciao.simulator.Simulator()
    sensor = ciao.sensors.Sensor(sim)
    mirror = sim
else:
    if ccfg.camera_id=='pylon':
        cam = ciao.cameras.PylonCamera()
    elif ccfg.camera_id=='ace':
        cam = ciao.cameras.AOCameraAce()
    else:
        sys.exit('Camera %s not available.'%ccfg.camera)
    mirror = ciao.mirrors.Mirror()
    sensor = ciao.sensors.Sensor(cam)
    
app = QApplication(sys.argv)
loop = ciao.loops.Loop(sensor,mirror)
ui = ciao.ui.UI(loop)
loop.start()
sys.exit(app.exec_())


