import ciao
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication

sim = ciao.simulator.Simulator()

sensor = ciao.sensors.Sensor(sim)
mirror = sim

app = QApplication(sys.argv)
loop = ciao.loops.Loop(sensor,mirror,verbose=3)
loop.load_poke('./etc_sim/ctrl/20190916142842_poke.txt')
loop.closed = True
loop.start()
sys.exit(app.exec_())


