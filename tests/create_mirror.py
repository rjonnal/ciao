import ciao
from matplotlib import pyplot as plt
import numpy as np

mirror = ciao.mirrors.Mirror()
mask = mirror.mask
command = np.zeros(mask.shape)
command[np.where(mask)] = mirror.controller.command
plt.figure()
plt.imshow(command)

    
plt.show()
