import numpy as np
import glob
from ciao import config as ccfg
import os,sys
try:
    from pypylon import pylon
except Exception as e:
    print e


class PylonCamera:

    def __init__(self,timeout=500):
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())

        self.camera.Open()

        # enable all chunks
        self.camera.ChunkModeActive = True
        #self.camera.PixelFormat = "Mono12"
        
        for cf in self.camera.ChunkSelector.Symbolics:
            self.camera.ChunkSelector = cf
            self.camera.ChunkEnable = True

        self.timeout = timeout

    def get_image(self):
        return self.camera.GrabOne(self.timeout).Array.astype(np.int16)


class SimulatedCamera:

    def __init__(self):
        self.image_list = sorted(glob.glob(os.path.join(ccfg.simulated_camera_image_directory,'*.npy')))
        self.n_images = len(self.image_list)
        self.index = 0
        #self.images = [np.load(fn) for fn in self.image_list]
        self.opacity = False
        self.sy,self.sx = np.load(self.image_list[0]).shape
        self.oy = int(round(np.random.rand()*self.sy//2+self.sy//4))
        self.ox = int(round(np.random.rand()*self.sx//2+self.sx//4))
        self.XX,self.YY = np.meshgrid(np.arange(self.sx),np.arange(self.sy))


    def set_opacity(self,val):
        self.opacity = val

    def get_opacity(self):
        return self.opacity
            
    def get_image(self):
        im = np.load(self.image_list[self.index])
        #im = self.images[self.index]

        if self.opacity:
            im = self.opacify(im)
            self.oy = self.oy+np.random.randn()*.5
            self.ox = self.ox+np.random.randn()*.5

        self.index = (self.index + 1)%self.n_images
        return im
        
    
    def opacify(self,im,sigma=50):
        xx,yy = self.XX-self.ox,self.YY-self.oy
        d = np.sqrt(xx**2+yy**2)
        #mask = np.exp((-d)/(2*sigma**2))
        #mask = mask/mask.max()
        #mask = 1-mask
        mask = np.ones(d.shape)
        mask[np.where(d<=sigma)] = 0.2
        out = np.round(im*mask).astype(np.int16)
        return out
