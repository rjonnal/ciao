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
from reference_generator import ReferenceGenerator
from ciao import config as ccfg
from frame_timer import FrameTimer
from poke import Poke

sensor_mutex = QMutex()
mirror_mutex = QMutex()

class Loop(QObject):

    finished = pyqtSignal()
    pause_signal = pyqtSignal()
    unpause_signal = pyqtSignal()
    
    def __init__(self,sensor,mirror,verbose=0):
        super(Loop,self).__init__()

        self.verbose = verbose
        
        self.mirror_thread = QThread()
        self.sensor_thread = QThread()

        self.sensor = sensor
        self.active_lenslets = np.ones(self.sensor.n_lenslets).astype(int)
        self.mirror = mirror

        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        #DEBUG
        self.sensor.moveToThread(self.sensor_thread)
        self.mirror.moveToThread(self.mirror_thread)

        # We have to connect the mirror timer's timeout signal
        # to the mirror update slot, and then start the timer
        # here. It's a little awkward, but the mirror timer
        # cannot be started until it's in its own thread, and
        # because we've used moveToThread (instead of
        # making Mirror a QThread subclass).
        self.mirror.timer.timeout.connect(self.mirror.update)
        self.mirror.timer.start(1.0/self.mirror.update_rate*1000.0)

        self.sensor_thread.started.connect(self.sensor.update)
        self.finished.connect(self.sensor.update)
        self.sensor.finished.connect(self.update)
        
        self.pause_signal.connect(self.sensor.pause)
        self.pause_signal.connect(self.mirror.pause)
        self.unpause_signal.connect(self.sensor.unpause)
        self.unpause_signal.connect(self.mirror.unpause)
        self.poke = None
        self.closed = False


        
        # try to load the poke file specified in
        # ciao_config.py; if it doesn't exist, create
        # a dummy poke with all 1's; this will result
        # in an inverse control matrix with very low
        # gains, i.e. the mirror won't be driven
        if not os.path.exists(ccfg.poke_filename):
            dummy = np.ones((2*n_lenslets,n_actuators))
            np.savetxt(ccfg.poke_filename,dummy)
            
        self.load_poke(ccfg.poke_filename)
        self.gain = ccfg.loop_gain
        self.loss = ccfg.loop_loss
        self.paused = False

    def has_poke(self):
        return self.poke is not None

    def start(self):
        if self.verbose>=5:
            print 'Starting loop.'
        self.sensor_thread.start()
        self.mirror_thread.start()

    def pause(self):
        self.pause_signal.emit()
        self.paused = True

    def unpause(self):
        self.unpause_signal.emit()
        self.paused = False
        self.finished.emit()
        print 'loop unpaused'
        
    @pyqtSlot()
    def update(self):
        if not self.paused:
            if self.verbose>=5:
                print 'Updating loop.'
            sensor_mutex.lock()
            mirror_mutex.lock()
            # compute the mirror command here
            if self.closed and self.has_poke():
                current_active_lenslets = np.zeros(self.active_lenslets.shape)
                current_active_lenslets[np.where(self.sensor.box_maxes>ccfg.spots_threshold)] = 1
                if not all(self.active_lenslets==current_active_lenslets):
                    self.active_lenslets[:] = current_active_lenslets[:]
                    self.poke.invert(mask=self.active_lenslets)

                xs = self.sensor.x_slopes[np.where(self.active_lenslets)[0]]
                ys = self.sensor.y_slopes[np.where(self.active_lenslets)[0]]
                if self.verbose>=1:
                    error = self.sensor.error
                    pcount = int(round(error*1e8))
                    print 'rms'+'.'*pcount
                
                slope_vec = np.hstack((xs,ys))
                command = self.gain * np.dot(self.poke.ctrl,slope_vec)
                command = self.mirror.get_command()*(1-self.loss) - command
                self.mirror.set_command(command)

                if self.verbose>=1:
                    if command.max()>ccfg.mirror_command_max*.95:
                        print 'actuator saturated'
                    if command.min()<ccfg.mirror_command_min*.95:
                        print 'actuator saturated'
                
            self.finished.emit()
            sensor_mutex.unlock()
            mirror_mutex.unlock()
                
    def load_poke(self,poke_filename=None):
        sensor_mutex.lock()
        mirror_mutex.lock()
        try:
            poke = np.loadtxt(poke_filename)
        except Exception as e:
            error_message('Could not find %s.'%poke_filename)
            options = QFileDialog.Options()
            #options |= QFileDialog.DontUseNativeDialog
            poke_filename, _ = QFileDialog.getOpenFileName(
                            None,
                            "Please select a poke file.",
                            ccfg.poke_directory,
                            "Text Files (*.txt)",
                            options=options)
            poke = np.loadtxt(poke_filename)

        py,px = poke.shape
        expected_py = self.sensor.n_lenslets*2
        expected_px = self.mirror.n_actuators
        dummy = np.ones((expected_py,expected_px))
        
        try:
            assert (py==expected_py and px==expected_px)
        except AssertionError as ae:
            error_message('Poke matrix has shape (%d,%d), but (%d,%d) was expected. Using dummy matrix.'%(py,px,expected_py,expected_px))
            poke = dummy
            
        self.poke = Poke(poke)

        sensor_mutex.unlock()
        mirror_mutex.unlock()

    def invert(self):
        if self.poke is not None:
            self.pause()
            time.sleep(1)
            self.poke.invert()
            time.sleep(1)
            QApplication.processEvents()
            self.unpause()
            time.sleep(1)

    def set_n_modes(self,n):
        try:
            self.poke.n_modes = n
        except Exception as e:
            print e

    def get_n_modes(self):
        out = -1
        try:
            out = self.poke.n_modes
        except Exception as e:
            print e
        return out

    def get_condition_number(self):
        out = -1
        try:
            out = self.poke.cutoff_cond
        except Exception as e:
            print e
        return out
            
    def run_poke(self):
        cmin = ccfg.poke_command_min
        cmax = ccfg.poke_command_max
        n_commands = ccfg.poke_n_command_steps
        commands = np.linspace(cmin,cmax,n_commands)

        self.pause()
        time.sleep(1)
        
        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        x_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        y_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        
        for k_actuator in range(n_actuators):
            self.mirror.flatten()
            for k_command in range(n_commands):
                cur = commands[k_command]
                print k_actuator,cur
                self.mirror.set_actuator(k_actuator,cur)
                QApplication.processEvents()
                time.sleep(.01)
                self.sensor.sense()
                sensor_mutex.lock()
                x_mat[:,k_actuator,k_command] = self.sensor.x_slopes
                y_mat[:,k_actuator,k_command] = self.sensor.y_slopes
                sensor_mutex.unlock()
                self.finished.emit()
        # print 'done'
        self.mirror.flatten()
        
        d_commands = np.mean(np.diff(commands))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_commands,axis=2)
        y_response = np.mean(d_y_mat/d_commands,axis=2)
        poke = np.vstack((x_response,y_response))
        ns = now_string()


        # After we make a new poke matrix, we will save it in
        # two files: an archive file that can be used to keep
        # track of old poke matrices, and the file specified
        # in the config file, e.g., 'poke.txt'.
        # The archive filename will use the time date string
        # generated above. This filename will also be used to
        # save the commands and the mirror mode chart PDF.
        
        poke_fn = ccfg.poke_filename
        archive_poke_fn = os.path.join(ccfg.poke_directory,'%s_poke.txt'%ns)
        archive_command_fn = os.path.join(ccfg.poke_directory,'%s_currents.txt'%ns)
        archive_chart_fn = os.path.join(ccfg.poke_directory,'%s_modes.pdf'%ns)
        
        np.savetxt(poke_fn,poke)
        np.savetxt(archive_poke_fn,poke)
        np.savetxt(archive_command_fn,commands)
        save_modes_chart(archive_chart_fn,poke,commands,self.mirror.mirror_mask)
        self.poke = Poke(poke)
        
        time.sleep(1)
        self.unpause()

    def set_closed(self,val):
        self.closed = val
