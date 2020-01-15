from ximea import xiapi
import numpy as np
import sys,os
from matplotlib import pyplot as plt

#create instance for first connected camera
cam = xiapi.Camera()

#start communication
#to open specific device, use:
#cam.open_device_by_SN('41305651')
#(open by serial number)
print('Opening first camera...')
cam.open_device()

#settings
cam.set_exposure(10000)
print('Exposure was set to %i us' %cam.get_exposure())

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

for i in range(1000):
    #get data and pass them from camera to img
    cam.get_image(img)

    #get raw data from camera
    #for Python2.x function returns string
    #for Python3.x function returns bytes
    data_raw = img.get_image_data_raw()

    arr = np.reshape(np.frombuffer(data_raw,dtype=np.uint8),(img.height,img.width))

    # the last line is all we need for a frame acquisition; everything after may
    # be useful for debugging, etc.
    
    plt.clf()
    plt.imshow(arr,cmap='gray')
    plt.colorbar()
    plt.pause(.5)

    #transform data to list
    data = list(data_raw)

    #print image data and metadata
    print('Image number: ' + str(i))
    print('Image width (pixels):  ' + str(img.width))
    print('Image height (pixels): ' + str(img.height))
    print('First 10 pixels: ' + str(data[:10]))
    print('\n')    

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

print('Done.')



