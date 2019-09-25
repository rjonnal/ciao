import ciao
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication
from time import sleep

cam = ciao.cameras.PylonCamera()
sensor = ciao.sensors.Sensor(cam)
mirror = ciao.mirrors.Mirror()


flat_x = np.zeros(sensor.n_lenslets)
flat_y = np.zeros(sensor.n_lenslets)
pushed_x = np.zeros(sensor.n_lenslets)
pushed_y = np.zeros(sensor.n_lenslets)
for k in range(mirror.n_actuators):
    mirror.flatten()
    sensor.sense()
    flat_x[:] = sensor.x_slopes[:]
    flat_y[:] = sensor.y_slopes[:]
    
    mirror.set_actuator(k,.5)
    sleep(.01)
    sensor.sense()
    pushed_x[:] = sensor.x_slopes[:]
    pushed_y[:] = sensor.y_slopes[:]

    
    x_mad = np.mean(np.abs(pushed_x-flat_x))
    y_mad = np.mean(np.abs(pushed_y-flat_y))
    
    plt.plot(k,x_mad,'ks')
    plt.plot(k,y_mad,'bo')
    plt.pause(.01)
    print 'actuator %d x MAD %0.3f'%(k,1e3*x_mad)
    print 'actuator %d y MAD %0.3f'%(k,1e3*y_mad)
    print

plt.show()