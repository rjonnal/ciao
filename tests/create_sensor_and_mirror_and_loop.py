import ciao
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication

#cam = ciao.cameras.SimulatedCamera()
cam = ciao.cameras.PylonCamera()
sensor = ciao.sensors.Sensor(cam)

sb = sensor.search_boxes
mirror = ciao.mirrors.Mirror()

app = QApplication(sys.argv)
loop = ciao.loops.Loop(sensor,mirror)
ui = ciao.ui.UI(loop)
loop.start()
sys.exit(app.exec_())


