import numpy as np
import time

class FrameTimer:
    def __init__(self,label,buffer_size=100,verbose=False):
        self.index = 0
        self.fps = 0.0
        self.frame_time = 0.0
        self.frame_rms = 0.0
        self.buff = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.label = label
        self.verbose = verbose
        
    def tick(self):
        self.buff[self.index] = time.time()
        self.index = self.index + 1
        if self.index==self.buffer_size:
            # buffer full--compute
            dt = np.diff(self.buff)
            self.frame_time = dt.mean()
            self.frame_rms = dt.std()
            self.fps = 1.0/self.frame_time
            self.index=0
            if self.verbose:
                print '%s: %0.1f (ms) %0.1f (ms std) %0.1f (fps)'%(self.label,1000.*self.frame_time,1000.*self.frame_rms,self.fps)

