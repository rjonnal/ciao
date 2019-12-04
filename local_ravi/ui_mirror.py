import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao
from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

mirror = ciao.mirrors.Mirror()
app = QApplication(sys.argv)
ui = ciao.ui.MirrorUI(mirror)
sys.exit(app.exec_())


