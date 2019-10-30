import random
import time
import sys, os
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QCheckBox, QHBoxLayout

N = 1000000

class Sensor:
    """A dummy sensor that generates N measurements with an average
    value of its gain."""
    def __init__(self):
        self.N = N
        self.gain = 1.0
        self.values = [self.gain]*self.N
        
    def step(self):
        for k in range(self.N):
            self.values[k] = random.random()*self.gain*2.0

class Actuator:
    """A dummy device that requires N values to actuate."""
    def __init__(self):
        self.N = N
        self.values = [0.0]*self.N

    def step(self,new_values):
        for k in range(self.N):
            self.values[k] = new_values[k]
            
    def monitor(self):
        sample_size = min(1000,self.N)
        return sum(self.values[:sample_size])/float(sample_size)
    

class Loop:
    """The control loop, with a sensor and an actuator."""
    def __init__(self):
        self.sensor = Sensor()
        self.actuator = Actuator()
        
    def step(self):
        self.sensor.step()
        self.actuator.step(self.sensor.values)


class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.loop = loop
        self.running = False

        self.label = QLabel('0.000')
        self.cb_running = QCheckBox('Run')
        self.cb_running.setChecked(False)
        self.cb_running.stateChanged.connect(self.set_running)
        
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.cb_running)
        
        self.setLayout(layout)
        self.show()
        while True:
            QApplication.processEvents()
            if self.running:
                self.step()

    def set_running(self,val):
        self.running = val
        print self.running

    def step(self):
        self.loop.step()
        self.label.setText('%0.3f'%self.loop.actuator.monitor())
        

if __name__ == '__main__':

    app = QApplication(sys.argv)
    loop = Loop()
    gui = UI(loop)
    sys.exit(app.exec_())
