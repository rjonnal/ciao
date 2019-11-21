import numpy as np
import time
import centroid
import sys
from PyQt5.QtCore import (QThread, QTimer, pyqtSignal, Qt, QPoint, QLine,
                          QMutex, QObject, pyqtSlot)

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage, QSlider)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import os
from matplotlib import pyplot as plt
import datetime
from tools import error_message, now_string, prepend, colortable, get_ram, get_process
import copy
from zernike import Reconstructor
import cProfile
import scipy.io as sio
from poke_analysis import save_modes_chart
from ctypes import CDLL,c_void_p
from search_boxes import SearchBoxes
import ciao_config as ccfg
from frame_timer import FrameTimer

class Sensor:

    def __init__(self,camera):
        self.image_width_px = ccfg.image_width_px
        self.image_height_px = ccfg.image_height_px
        self.dark_image = np.zeros((ccfg.image_height_px,ccfg.image_width_px),dtype=np.int16)
        self.dark_subtract = False
        self.n_dark = 10
        self.lenslet_pitch_m = ccfg.lenslet_pitch_m
        self.lenslet_focal_length_m = ccfg.lenslet_focal_length_m
        self.pixel_size_m = ccfg.pixel_size_m
        self.beam_diameter_m = ccfg.beam_diameter_m
        self.wavelength_m = ccfg.wavelength_m
        self.background_correction = ccfg.background_correction
        self.centroiding_iterations = ccfg.centroiding_iterations
        self.iterative_centroiding_step = ccfg.iterative_centroiding_step
        self.filter_lenslets = ccfg.sensor_filter_lenslets
        self.estimate_background = ccfg.estimate_background
        self.reconstruct_wavefront = ccfg.sensor_reconstruct_wavefront
        self.remove_tip_tilt = ccfg.sensor_remove_tip_tilt

        # calculate diffraction limited spot size on sensor, to determine
        # the centroiding window size
        lenslet_dlss = 1.22*self.wavelength_m*self.lenslet_focal_length_m/self.lenslet_pitch_m
        lenslet_dlss_px = lenslet_dlss/self.pixel_size_m
        # now we need to account for smearing of spots due to axial displacement of retinal layers
        extent = 500e-6 # retinal thickness
        smear = extent*6.75/16.67 # pupil diameter and focal length; use diameter in case beacon is at edge of field
        try:
            magnification = ccfg.retina_sensor_magnification
        except Exception as e:
            magnification = 1.0
        total_size = lenslet_dlss+smear*magnification
        total_size_px = total_size/self.pixel_size_m
        self.centroiding_half_width = int(np.floor(total_size_px/2.0))*2
        
        try:
            xy = np.loadtxt(ccfg.reference_coordinates_filename)
        except Exception as e:
            xy = np.loadtxt(ccfg.reference_coordinates_bootstrap_filename)
            print 'Bootstrapping with %s'%ccfg.reference_coordinates_bootstrap_filename
            
        self.search_boxes = SearchBoxes(xy[:,0],xy[:,1],ccfg.search_box_half_width)
        self.sensor_mask = np.loadtxt(ccfg.reference_mask_filename)
        
        self.x0 = np.zeros(self.search_boxes.x.shape)
        self.y0 = np.zeros(self.search_boxes.y.shape)
        
        self.x0[:] = self.search_boxes.x[:]
        self.y0[:] = self.search_boxes.y[:]
        
        self.n_lenslets = self.search_boxes.n
        n_lenslets = self.n_lenslets
        self.image = np.zeros((ccfg.image_height_px,ccfg.image_width_px))
        self.x_slopes = np.zeros(n_lenslets)
        self.y_slopes = np.zeros(n_lenslets)
        self.x_centroids = np.zeros(n_lenslets)
        self.y_centroids = np.zeros(n_lenslets)
        self.box_maxes = np.zeros(n_lenslets)
        self.box_mins = np.zeros(n_lenslets)
        self.box_means = np.zeros(n_lenslets)
        self.valid_centroids = np.zeros(n_lenslets,dtype=np.int16)

        try:
            self.fast_centroiding = ccfg.fast_centroiding
        except Exception as e:
            self.fast_centroiding = False
        self.box_backgrounds = np.zeros(n_lenslets)
        self.error = 0.0
        self.tip = 0.0
        self.tilt = 0.0
        self.zernikes = np.zeros(ccfg.n_zernike_terms)
        self.wavefront = np.zeros(self.sensor_mask.shape)

        self.image_max = -1
        self.image_min = -1
        self.image_mean = -1
        
        self.cam = camera
        self.frame_timer = FrameTimer('Sensor',verbose=False)
        self.reconstructor = Reconstructor(self.search_boxes.x,
                                           self.search_boxes.y,self.sensor_mask)
        self.logging = False
        self.paused = False
        
    def update(self):
        if not self.paused:
            try:
                self.sense()
            except Exception as e:
                print 'Sensor update exception:',e
            if self.logging:
                self.log()
                
        self.frame_timer.tick()
    
    def pause(self):
        print 'sensor paused'
        self.paused = True

    def unpause(self):
        print 'sensor unpaused'
        self.paused = False

    def set_dark_subtraction(self,val):
        self.dark_subtract = val

    def set_n_dark(self,val):
        self.n_dark = val

    def set_dark(self):
        self.pause()
        temp = np.zeros(self.dark_image.shape)
        for k in range(self.n_dark):
            temp = temp + self.cam.get_image()
        temp = np.round(temp/float(self.n_dark)).astype(np.int16)
        self.dark_image[...] = temp[...]
        self.unpause()

    def log(self):
        outfn = os.path.join(ccfg.logging_directory,'sensor_%s.mat'%(now_string(True)))
        d = {}
        d['x_slopes'] = self.x_slopes
        d['y_slopes'] = self.y_slopes
        d['x_centroids'] = self.x_centroids
        d['y_centroids'] = self.y_centroids
        d['search_box_x1'] = self.search_boxes.x1
        d['search_box_x2'] = self.search_boxes.x2
        d['search_box_y1'] = self.search_boxes.y1
        d['search_box_y2'] = self.search_boxes.y2
        d['ref_x'] = self.search_boxes.x
        d['ref_y'] = self.search_boxes.y
        d['error'] = self.error
        d['tip'] = self.tip
        d['tilt'] = self.tilt
        d['wavefront'] = self.wavefront
        d['zernikes'] = self.zernikes
        
        sio.savemat(outfn,d)

    def set_background_correction(self,val):
        #sensor_mutex.lock()
        self.background_correction = val
        #sensor_mutex.unlock()

    def set_fast_centroiding(self,val):
        self.fast_centroiding = val

    def set_logging(self,val):
        self.logging = val


    def set_defocus(self,val):
        self.pause()
        
        newx = self.x0 + self.reconstructor.defocus_dx*val*ccfg.zernike_dioptric_equivalent
        
        newy = self.y0 + self.reconstructor.defocus_dy*val*ccfg.zernike_dioptric_equivalent
        self.search_boxes.move(newx,newy)
        
        self.unpause()

    def set_centroiding_half_width(self,val):
        self.centroiding_half_width = val
        
    def sense(self,debug=False):
        self.image = self.cam.get_image()
        if self.dark_subtract:
            self.image = self.image - self.dark_image

        self.image_min = self.image.min()
        self.image_mean = self.image.mean()
        self.image_max = self.image.max()
        
        t0 = time.time()
        if not self.fast_centroiding:
            centroid.estimate_backgrounds(spots_image=self.image,
                                          sb_x_vec = self.search_boxes.x,
                                          sb_y_vec = self.search_boxes.y,
                                          sb_bg_vec = self.box_backgrounds,
                                          sb_half_width_p = self.search_boxes.half_width)
            centroid.compute_centroids(spots_image=self.image,
                                       sb_x_vec = self.search_boxes.x,
                                       sb_y_vec = self.search_boxes.y,
                                       sb_bg_vec = self.box_backgrounds,
                                       sb_half_width_p = self.search_boxes.half_width,
                                       iterations_p = self.centroiding_iterations,
                                       iteration_step_px_p = self.iterative_centroiding_step,
                                       x_out = self.x_centroids,
                                       y_out = self.y_centroids,
                                       mean_intensity = self.box_means,
                                       maximum_intensity = self.box_maxes,
                                       minimum_intensity = self.box_mins,
                                       num_threads_p = 1)
        else:
            centroid.fast_centroids(spots_image=self.image,
                                    sb_x_vec = self.search_boxes.x,
                                    sb_y_vec = self.search_boxes.y,
                                    sb_half_width_p = self.search_boxes.half_width,
                                    centroiding_half_width_p = self.centroiding_half_width,
                                    x_out = self.x_centroids,
                                    y_out = self.y_centroids,
                                    sb_max_vec = self.box_maxes,
                                    valid_vec = self.valid_centroids,
                                    verbose_p = 0,
                                    num_threads_p = 1)
        self.centroiding_time = time.time()-t0
        self.x_slopes = (self.x_centroids-self.search_boxes.x)*self.pixel_size_m/self.lenslet_focal_length_m
        self.y_slopes = (self.y_centroids-self.search_boxes.y)*self.pixel_size_m/self.lenslet_focal_length_m
        self.tilt = np.mean(self.x_slopes)
        self.tip = np.mean(self.y_slopes)
        if self.remove_tip_tilt:
            self.x_slopes-=self.tilt
            self.y_slopes-=self.tip
        if self.reconstruct_wavefront:
            self.zernikes,self.wavefront,self.error = self.reconstructor.get_wavefront(self.x_slopes,self.y_slopes)
        
        
    def record_reference(self):
        print 'recording reference'
        self.pause()
        xcent = []
        ycent = []
        for k in range(ccfg.reference_n_measurements):
            print 'measurement %d of %d'%(k+1,ccfg.reference_n_measurements),
            self.sense()
            print '...done'
            xcent.append(self.x_centroids)
            ycent.append(self.y_centroids)

        xcent = np.array(xcent)
        ycent = np.array(ycent)
        
        x_ref = xcent.mean(0)
        y_ref = ycent.mean(0)

        x_var = xcent.var(0)
        y_var = ycent.var(0)
        err = np.sqrt(np.mean([x_var,y_var]))
        print 'reference coordinate error %0.3e px RMS'%err
        
        self.search_boxes = SearchBoxes(x_ref,y_ref,self.search_boxes.half_width)
        refxy = np.array((x_ref,y_ref)).T

        # Record the new reference set to two locations, the
        # filename specified by reference_coordinates_filename
        # in ciao config, and also an archival filename to keep
        # track of the history.
        archive_fn = os.path.join(ccfg.reference_directory,prepend('reference.txt',now_string()))
        
        np.savetxt(archive_fn,refxy,fmt='%0.3f')
        np.savetxt(ccfg.reference_coordinates_filename,refxy,fmt='%0.3f')
        
        self.unpause()
        time.sleep(1)
