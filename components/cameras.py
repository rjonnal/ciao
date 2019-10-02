import numpy as np
import glob
from ciao import config as ccfg
import os,sys
try:
    from pypylon import pylon
except Exception as e:
    print e
from ctypes import *
from ctypes.util import find_library
import milc
from time import time

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

    def close(self):
        return

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
        
    def close(self):
        return
        

class AOCameraAce():

    _MilImage0 = c_longlong(0)
    _MilImage1 = c_longlong(0)
    _InitFlag = c_longlong(milc.M_PARTIAL)
    #_InitFlagD = c_longlong(milc.M_DEFAULT)


    _MilApplication = c_longlong()
    _MilSystem = c_longlong()
    _MilDigitizer = c_longlong()
    _MilImage0 = c_longlong()
    _MilImage1 = c_longlong()

    
    def __init__(self):
    
        if sys.platform=='win32':
            self._mil = windll.LoadLibrary("mil")
        else:
            sys.exit('pyao.cameras assumes Windows DLL shared libraries')
            
        self._cameraFilename = os.path.join(ccfg.dcf_directory,'acA2040-180km-4tap-12bit_reloaded.dcf')

        # a quick fix to a deep problem; force mode 0; it slows things down a bit
        # but guards against the memory management issue
        self._mode = 2

        self._mil.MappAllocW.argtypes = [c_longlong, POINTER(c_longlong)] 
        self._mil.MsysAllocW.argtypes = [c_wchar_p, c_longlong, c_longlong, POINTER(c_longlong)]
        self._mil.MdigAllocW.argtypes = [c_longlong, c_longlong, c_wchar_p, c_longlong, POINTER(c_longlong)]
        self._mil.MbufAllocColor.argtypes = [c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, c_longlong, POINTER(c_longlong)]

        self._mil.MappAllocW(self._InitFlag,byref(self._MilApplication))
        self.printMilError()
        print 'MIL App Identifier: %d'%self._MilApplication.value

        cSystemName = c_wchar_p('M_SYSTEM_SOLIOS')
        self._mil.MsysAllocW(cSystemName, milc.M_DEFAULT, 
                           self._InitFlag, byref(self._MilSystem))
        self.printMilError()
        print 'MIL Sys Identifier: %d'%self._MilSystem.value

        
        cDcfFn = c_wchar_p(self._cameraFilename)

        self._mil.MdigAllocW(self._MilSystem, milc.M_DEFAULT, cDcfFn, milc.M_DEFAULT, byref(self._MilDigitizer))
        self.printMilError()
        print 'MIL Dig Identifier: %d'%self._MilDigitizer.value

        nBands = c_longlong(1)
        bufferType = c_longlong(milc.M_SIGNED + 16)
        bufferAttribute = c_longlong(milc.M_GRAB + milc.M_IMAGE)
        
        binning = 1
        self.n_sig = 10
        self._xSizePx = 2048
        self._ySizePx = 2048
        #print binning
        #sys.exit()

        if self._mode==0: # double buffer / continuous grab
            self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
            self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage1))
            self.printMilError()
            print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value), self._mode
            self._mil.MdigGrabContinuous(self._MilDigitizer,self._MilImage0)
            self.printMilError()
        elif self._mode==1: # double buffer / single grabs
            self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
            self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage1))
            self.printMilError()
            print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value), self._mode
            self._mil.MdigControlInt64(self._MilDigitizer,milc.M_GRAB_MODE,milc.M_SYNCHRONOUS)
            self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
        elif self._mode==2: # single buffer / single grabs
            self._mil.MbufAllocColor(self._MilSystem, nBands, self._xSizePx, self._ySizePx, bufferType, bufferAttribute, byref(self._MilImage0))
            self.printMilError()
            print 'MIL Img Identifiers: %d,%d'%(self._MilImage0.value, self._MilImage1.value), self._mode
            # changing the GRAB_MODE from ASYNCHRONOUS to SYNCHRONOUS seemed to improve the slope responses during the poke matrix
            # acquisition
            self._mil.MdigControlInt64(self._MilDigitizer,milc.M_GRAB_MODE,milc.M_SYNCHRONOUS)
            self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
            
        self._im = np.zeros([np.int16(self._ySizePx),np.int16(self._xSizePx)]).astype(np.int16)
        self._im_ptr = self._im.ctypes.data


        

        # there must be a bug in camera initialization code above because if the camera has been
        # sitting, on, for a while, the first few frames (up to 3, anecdotally) may have serious
        # geometry problems (via, it seems, tap reordering). A quick and dirty fix: grab a few images
        # upon initialization:
        nBad = 5
        for iBad in range(nBad):
            bad = self.getImage()


    def close(self):
        print 'Closing camera...'
        self._mil.MdigHalt(self._MilDigitizer)
        self.printMilError()
        self._mil.MbufFree(self._MilImage0)
        self.printMilError()
        self._mil.MbufFree(self._MilImage1)
        self.printMilError()
        self._mil.MdigFree(self._MilDigitizer)
        self.printMilError()
        self._mil.MsysFree(self._MilSystem)
        self.printMilError()
        self._mil.MappFree(self._MilApplication)

    def getSignature(self):
        N = self.n_sig
        return '_'.join(['%d'%p for p in [self._im[500,50+k] for k in range(N)]])
        
    def getImage(self):
        self.updateImage()
        return self._im
        
    def get_image(self):
        return self.getImage()
        
    def updateImage(self):
        t0 = time()
        sig = self.getSignature()
        done = False
        count = 0
        while not done:
            #print 'wait count',count,'pixels',self._im[200,200:205]
            if self._mode==0:
                # sleeping does work here (100 ms)
                #self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END );
                # sleeping does work here (100 ms)
                self._mil.MbufCopy(self._MilImage0,self._MilImage1)
                # sleeping doesn't work here (100 ms)
                self._mil.MbufGet(self._MilImage1,self._im_ptr)
            elif self._mode==1:
                self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END )
                self._mil.MbufCopy(self._MilImage0,self._MilImage1)
                self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
                self._mil.MbufGet(self._MilImage1,self._im_ptr)
            elif self._mode==2:
                #t1 = time()
                self._mil.MdigGrabWait(self._MilDigitizer, milc.M_GRAB_FRAME_END )
                #t2 = time()
                #print 'MdigGrabWait took %0.4f s'%(t2-t1)
                self._mil.MdigGrab(self._MilDigitizer,self._MilImage0)
                self._mil.MbufGet(self._MilImage0,self._im_ptr)
            new_sig = self.getSignature()
            done = (not sig==new_sig) or (new_sig==('0_'*self.n_sig)[:-1])
            count+=1
        t1 = time()
            
    def printMilError(self):
        err = c_longlong(0)
        self._mil.MappGetError(2L,byref(err))
        print 'MIL Error Code: %d'%err.value
    

