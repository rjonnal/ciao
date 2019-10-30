import random
import time
import sys, os
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QCheckBox, QHBoxLayout, QPushButton, QDoubleSpinBox

N = 100000

class SensorState(QObject):
    def __init__(self,L):
        super(SensorState,self).__init__()
        self.L = L
        
    def set_state(self,L):
        self.L = L


class SensorConfiguration(QObject):
    def __init__(self):
        super(SensorConfiguration,self).__init__()
        self.gain = 1.0
        
class Sensor(QObject):
    """A dummy sensor that generates N measurements with an average
    value of its gain."""

    finished = pyqtSignal(QObject)
    
    def __init__(self):
        super(Sensor,self).__init__()
        self.N = N
        self.cfg = SensorConfiguration()
        self.values = [self.cfg.gain]*self.N
        self.state = SensorState(self.values)

    @pyqtSlot()
    def step(self):
        for k in range(self.N):
            self.values[k] = random.random()*self.cfg.gain*2.0
        self.finished.emit(self.state)

    @pyqtSlot(QObject)
    def set_configuration(self,cfg):
        self.cfg = cfg
        
        
class ActuatorState(QObject):
    def __init__(self,L):
        super(ActuatorState,self).__init__()
        self.L = L

    def set_state(self,L):
        self.L = L
        
class Actuator(QObject):
    actuated = pyqtSignal(QObject)
    
    """A dummy device that requires N values to actuate."""
    def __init__(self):
        super(Actuator,self).__init__()
        self.N = N
        self.values = [0]*self.N
        self.timer = QTimer()
        self.timer.timeout.connect(self.step)
        self.timer.start(10.)
        self.state = ActuatorState(self.values)

    def step(self):
        self.actuated.emit(self.state)
        
    def set(self,new_values):
        for k in range(self.N):
            self.values[k] = int(round(new_values[k]))
            
    def monitor(self):
        sample_size = min(1000,self.N)
        return sum(self.values[:sample_size])/float(sample_size)
    

class Loop(QObject):
    finished = pyqtSignal()
    
    """The control loop, with a sensor and an actuator."""
    def __init__(self):
        super(Loop,self).__init__()
        self.sensor = Sensor()
        
        self.actuator = Actuator()
        
        self.sensor_thread = QThread()
        self.actuator_thread = QThread()
        
        self.sensor.moveToThread(self.sensor_thread)
        self.actuator.moveToThread(self.actuator_thread)
        
        self.sensor_thread.started.connect(self.sensor.step)
        self.finished.connect(self.sensor.step)
        self.sensor.finished.connect(self.update)
        self.actuator.actuated.connect(self.monitor_actuator)

    def start(self):
        self.sensor_thread.start()
        self.actuator_thread.start()

    @pyqtSlot(QObject)
    def update(self,sensor_state):
        #QApplication.processEvents()
        self.actuator.set(self.sensor.values)
        self.finished.emit()

    @pyqtSlot(QObject)
    def monitor_actuator(self,actuator_state):
        print actuator_state.L[0]
        
    def quit(self):
        self.sensor_thread.threadactive = False
        #self.sensor_thread.wait()
        self.actuator_thread.threadactive = False
        #self.actuator_thread.wait()
        
class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.loop = loop
        self.running = False

        self.label = QLabel('0.000')

        self.cb_running = QCheckBox('Run')
        self.cb_running.setChecked(False)
        self.cb_running.stateChanged.connect(self.set_running)

        self.sb_gain = QDoubleSpinBox()
        self.sb_gain.setMinimum(0.1)
        self.sb_gain.setMaximum(5.0)
        self.sb_gain.setValue(1.0)
        self.sb_gain.setSingleStep(0.1)
        self.sb_gain.valueChanged[float].connect(self.update_gain)
        
        self.pb_quit = QPushButton('Quit')
        self.pb_quit.clicked.connect(self.quit)
        
        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.cb_running)
        layout.addWidget(self.sb_gain)
        layout.addWidget(self.pb_quit)
        
        self.setLayout(layout)
        #self.show()
        #while True:
        #    QApplication.processEvents()
        #    if self.running:
        #        self.step()

    def update_gain(self,val):
        print val
        
    def set_running(self,val):
        self.running = val
        print self.running

    def step(self):
        self.loop.step()
        self.label.setText('%0.3f'%self.loop.actuator.monitor())

    @pyqtSlot(QObject)
    def update_sensor_configuration(self,cfg):
        self.sensor_configuration = cfg

    def quit(self):
        self.loop.quit()
        sys.exit()

    
        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    loop = Loop()
    gui = UI(loop)
    gui.show()
    sys.exit(app.exec_())
