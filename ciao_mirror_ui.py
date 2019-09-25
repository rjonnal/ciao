import ciao
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication

#mirror = ciao.mirrors.Mirror()
mirror = ciao.simulator.Simulator()
app = QApplication(sys.argv)
ui = ciao.ui.MirrorUI(mirror)
sys.exit(app.exec_())


