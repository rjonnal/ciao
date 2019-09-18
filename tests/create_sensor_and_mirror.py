import ciao
from matplotlib import pyplot as plt
import numpy as np

cam = ciao.cameras.SimulatedCamera()
sensor = ciao.sensors.Sensor(cam)

sensor.sense()


sb = sensor.search_boxes
plt.figure()
plt.imshow(sensor.image,cmap='gray',aspect='auto')
for x1,x2,y1,y2 in zip(sb.x1,sb.x2,sb.y1,sb.y2):
    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'y-')

mirror = ciao.mirrors.Mirror()
mask = mirror.mask
command = np.zeros(mask.shape)
command[np.where(mask)] = mirror.controller.command
plt.figure()
plt.imshow(command)

    
plt.show()
