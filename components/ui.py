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
                             QErrorMessage, QSlider, QGraphicsView)
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
import os

#sensor_mutex = QMutex()
#mirror_mutex = QMutex()

try:
    os.mkdir('.gui_settings')
except Exception as e:
    print e


class Overlay(QWidget):
    """Stores a list of 4-tuples (x1,x2,y1,y2), and can draw
    these over its pixmap as either lines between (x1,y1) and (x2,y2),
    or rects [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]."""
    
    def __init__(self,coords=[],color=(255,255,255,255),thickness=1.0,mode='rects',visible=True):
        self.coords = coords
        self.pen = QPen()
        self.pen.setColor(QColor(*color))
        self.pen.setWidth(thickness)
        self.mode = mode
        self.visible = visible
        
    def draw(self,pixmap,downsample=1):
        d = float(downsample)
        if not self.visible:
            return
        if self.mode=='lines':
            painter = QPainter()
            painter.begin(pixmap)
            painter.setPen(self.pen)
            for index,(x1,x2,y1,y2) in enumerate(self.coords):
                painter.drawLine(QLine(x1/d,y1/d,x2/d,y2/d))
            painter.end()
        elif self.mode=='rects':
            painter = QPainter()
            painter.begin(pixmap)
            painter.setPen(self.pen)
            for index,(x1,x2,y1,y2) in enumerate(self.coords):
                width = x2-x1
                height = y2-y1
                painter.drawRect(x1/d,y1/d,width/d,height/d)
            painter.end()
            
    
    
class ZoomDisplay(QWidget):
    def __init__(self,name,clim=(0,255),colormap='gray',zoom=1.0,overlays=[],downsample=1):
        super(ZoomDisplay,self).__init__()
        self.name = name
        self.clim = clim
        self.zoom = zoom
        self.pixmap = QPixmap()
        self.label = QLabel()
        self.overlays = overlays
        self.colormap = colormap
        self.colortable = colortable(self.colormap)
        self.sized = False
        self.display_ratio = 1.0
        self.downsample = downsample
        
        self.mouse_coords = (ccfg.zoom_width/2.0,ccfg.zoom_height/2.0)
        self.sy,self.sx = 256,256 #initialize to something random
        
        layout = QHBoxLayout()
        layout.addWidget(self.label)

        # set up contrast sliders:
        self.n_steps = 20
        self.cmin_slider = QSlider(Qt.Vertical)
        self.cmax_slider = QSlider(Qt.Vertical)

        self.cmin_slider.setMinimum(0)
        self.cmax_slider.setMinimum(0)

        self.cmin_slider.setSingleStep(1)
        self.cmax_slider.setSingleStep(1)

        self.cmin_slider.setPageStep(5)
        self.cmax_slider.setPageStep(5)

        self.cmin_slider.setMaximum(self.n_steps)
        self.cmax_slider.setMaximum(self.n_steps)

        try:
            self.display_clim = np.loadtxt(os.path.join('.gui_settings','clim_%s.txt'%self.name))
        except Exception as e:
            print e
            self.display_clim = self.clim
            
        self.cmax_slider.setToolTip('%0.1e'%self.display_clim[1])
        self.cmin_slider.setToolTip('%0.1e'%self.display_clim[0])

        self.cmin_slider.setValue(self.real2slider(self.display_clim[0]))
        self.cmax_slider.setValue(self.real2slider(self.display_clim[1]))
        self.cmin_slider.valueChanged.connect(self.set_cmin)
        self.cmax_slider.valueChanged.connect(self.set_cmax)

        layout.addWidget(self.cmin_slider)
        layout.addWidget(self.cmax_slider)
        self.setLayout(layout)
        
    def mousePressEvent(self,e):
        self.mouse_coords = (e.x()*self.display_ratio,e.y()*self.display_ratio)
        
    def zoomed(self):
        x1 = int(round(self.mouse_coords[0]-ccfg.zoom_width//2))
        x2 = int(round(self.mouse_coords[0]+ccfg.zoom_width//2))
        if x1<0:
            x2 = x2 - x1
            x1 = 0
        if x2>=self.sx:
            x1 = x1 - (x2-self.sx) - 1
            x2 = self.sx - 1
            
        y1 = int(round(self.mouse_coords[1]-ccfg.zoom_height//2))
        y2 = int(round(self.mouse_coords[1]+ccfg.zoom_height//2))
        if y1<0:
            y2 = y2 - y1
            y1 = 0
        
        if y2>=self.sy:
            y1 = y1 - (y2-self.sy) - 1
            y2 = self.sy - 1

        return self.data[y1:y2,x1:x2]
        
    def real2slider(self,val):
        # convert a real value into a slider value
        return round(int((val-float(self.clim[0]))/float(self.clim[1]-self.clim[0])*self.n_steps))

    def slider2real(self,val):
        # convert a slider integer into a real value
        return float(val)/float(self.n_steps)*(self.clim[1]-self.clim[0])+self.clim[0]
    
    def set_cmax(self,slider_value):
        self.display_clim = (self.display_clim[0],self.slider2real(slider_value))
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.display_clim)
        self.cmax_slider.setToolTip('%0.1e'%self.display_clim[1])
        
    def set_cmin(self,slider_value):
        self.display_clim = (self.slider2real(slider_value),self.display_clim[1])
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.display_clim)
        self.cmin_slider.setToolTip('%0.1e'%self.display_clim[0])

    def show(self,data):
        data = data[::self.downsample,::self.downsample]
        self.data = data
        if not self.sized:
            self.label.setMinimumHeight(int(self.zoom*data.shape[0]))
            self.label.setMinimumWidth(int(self.zoom*data.shape[1]))
            self.sized = True
        try:
            cmin,cmax = self.display_clim
            bmp = np.round(np.clip((data.astype(np.float)-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
            self.sy,self.sx = bmp.shape
            n_bytes = bmp.nbytes
            bytes_per_line = int(n_bytes/self.sy)
            image = QImage(bmp,self.sy,self.sx,bytes_per_line,QImage.Format_Indexed8)
            image.setColorTable(self.colortable)
            self.pixmap.convertFromImage(image)
            for o in self.overlays:
                o.draw(self.pixmap,self.downsample)
            #self.label.setPixmap(self.pixmap)
            self.label.setPixmap(self.pixmap.scaled(self.label.width(),self.label.height(),Qt.KeepAspectRatio))
            self.display_ratio = float(self.sx)/float(self.label.width())
        except Exception as e:
            print e

class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.sensor_mutex = QMutex()#loop.sensor_mutex
        self.mirror_mutex = QMutex()#loop.mirror_mutex
        self.loop = loop
        self.loop.finished.connect(self.update)
        self.draw_boxes = ccfg.show_search_boxes
        self.draw_lines = ccfg.show_slope_lines
        self.init_UI()
        self.frame_timer = FrameTimer('UI',verbose=False)
        self.show()

    #def get_draw_boxes(self):
    #    return self.draw_boxes


    def keyPressEvent(self,event):
        if event.key()==Qt.Key_W:
            self.loop.sensor.search_boxes.up()
        if event.key()==Qt.Key_Z:
            self.loop.sensor.search_boxes.down()
        if event.key()==Qt.Key_A:
            self.loop.sensor.search_boxes.left()
        if event.key()==Qt.Key_S:
            self.loop.sensor.search_boxes.right()
        self.update_box_coords()
        
    def update_box_coords(self):
        self.boxes_coords = []
        for x1,x2,y1,y2 in zip(self.loop.sensor.search_boxes.x1,
                               self.loop.sensor.search_boxes.x2,
                               self.loop.sensor.search_boxes.y1,
                               self.loop.sensor.search_boxes.y2):
            self.boxes_coords.append((x1,x2,y1,y2))
            self.overlay_boxes.coords = self.boxes_coords
    
        
    def set_draw_boxes(self,val):
        self.draw_boxes = val
        self.overlay_boxes.visible = val
    #def get_draw_lines(self):
    #    return self.draw_lines

    def set_draw_lines(self,val):
        self.draw_lines = val
        self.overlay_slopes.visible = val

    def init_UI(self):
        self.setWindowIcon(QIcon('./icons/ciao.png'))
        self.setWindowTitle('CIAO')
        
        layout = QGridLayout()
        imax = 2**ccfg.bit_depth-1
        imin = 0

        self.boxes_coords = []
        for x1,x2,y1,y2 in zip(self.loop.sensor.search_boxes.x1,
                               self.loop.sensor.search_boxes.x2,
                               self.loop.sensor.search_boxes.y1,
                               self.loop.sensor.search_boxes.y2):
            self.boxes_coords.append((x1,x2,y1,y2))

        self.overlay_boxes = Overlay(coords=self.boxes_coords,mode='rects',color=ccfg.slope_line_color,thickness=ccfg.slope_line_thickness)

        self.overlay_slopes = Overlay(coords=[],mode='lines',color=ccfg.active_search_box_color,thickness=ccfg.slope_line_thickness)
        
        self.id_spots = ZoomDisplay('spots',clim=(0,4095),colormap=ccfg.spots_colormap,zoom=0.25,overlays=[self.overlay_boxes,self.overlay_slopes],downsample=2)
        
        layout.addWidget(self.id_spots,0,0,3,3)

        self.id_mirror = ZoomDisplay('mirror',clim=ccfg.mirror_clim,colormap=ccfg.mirror_colormap,zoom=30.0)
        self.id_wavefront = ZoomDisplay('wavefront',clim=ccfg.wavefront_clim,colormap=ccfg.wavefront_colormap,zoom=10.0)

        self.id_zoomed_spots = ZoomDisplay('zoomed_spots',clim=(0,4095),colormap=ccfg.spots_colormap,zoom=5.0)
        
        layout.addWidget(self.id_mirror,0,3,1,1)
        layout.addWidget(self.id_wavefront,1,3,1,1)
        layout.addWidget(self.id_zoomed_spots,2,3,1,1)
        
        column_2 = QVBoxLayout()
        column_2.setAlignment(Qt.AlignTop)
        self.cb_closed = QCheckBox('Loop &closed')
        self.cb_closed.setChecked(self.loop.closed)
        self.cb_closed.stateChanged.connect(self.loop.set_closed)

        self.cb_draw_boxes = QCheckBox('Draw boxes')
        self.cb_draw_boxes.setChecked(self.draw_boxes)
        self.cb_draw_boxes.stateChanged.connect(self.set_draw_boxes)

        self.cb_draw_lines = QCheckBox('Draw lines')
        self.cb_draw_lines.setChecked(self.draw_lines)
        self.cb_draw_lines.stateChanged.connect(self.set_draw_lines)

        self.cb_logging = QCheckBox('Logging')
        self.cb_logging.setChecked(False)
        self.cb_logging.stateChanged.connect(self.loop.sensor.set_logging)
        self.cb_logging.stateChanged.connect(self.loop.mirror.set_logging)
        
        self.pb_poke = QPushButton('Poke')
        self.pb_poke.clicked.connect(self.loop.run_poke)
        self.pb_record_reference = QPushButton('Record reference')
        self.pb_record_reference.clicked.connect(self.loop.sensor.record_reference)
        self.pb_flatten = QPushButton('&Flatten')
        self.pb_flatten.clicked.connect(self.flatten)

        
        self.pb_quit = QPushButton('&Quit')
        self.pb_quit.clicked.connect(sys.exit)

        poke_layout = QHBoxLayout()
        poke_layout.addWidget(QLabel('Modes:'))
        self.modes_spinbox = QSpinBox()
        self.modes_spinbox.setMaximum(ccfg.mirror_n_actuators)
        self.modes_spinbox.setMinimum(0)
        self.modes_spinbox.valueChanged.connect(self.loop.set_n_modes)
        self.modes_spinbox.setValue(self.loop.get_n_modes())
        poke_layout.addWidget(self.modes_spinbox)
        self.pb_invert = QPushButton('Invert')
        self.pb_invert.clicked.connect(self.loop.invert)
        poke_layout.addWidget(self.pb_invert)

        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel('Background correction:'))
        self.bg_spinbox = QSpinBox()
        self.bg_spinbox.setValue(self.loop.sensor.background_correction)
        self.bg_spinbox.setMaximum(500)
        self.bg_spinbox.setMinimum(-500)
        self.bg_spinbox.valueChanged.connect(self.loop.sensor.set_background_correction)
        bg_layout.addWidget(self.bg_spinbox)


        f_layout = QHBoxLayout()
        f_layout.addWidget(QLabel('Defocus:'))
        self.f_spinbox = QDoubleSpinBox()
        self.f_spinbox.setValue(0.0)
        self.f_spinbox.setSingleStep(0.01)
        self.f_spinbox.setMaximum(1.0)
        self.f_spinbox.setMinimum(-1.0)
        self.f_spinbox.valueChanged.connect(self.loop.sensor.set_defocus)
        f_layout.addWidget(self.f_spinbox)
        
        self.lbl_error = QLabel()
        self.lbl_error.setAlignment(Qt.AlignRight)
        self.lbl_tip = QLabel()
        self.lbl_tip.setAlignment(Qt.AlignRight)
        self.lbl_tilt = QLabel()
        self.lbl_tilt.setAlignment(Qt.AlignRight)
        self.lbl_cond = QLabel()
        self.lbl_cond.setAlignment(Qt.AlignRight)
        self.lbl_sensor_fps = QLabel()
        self.lbl_sensor_fps.setAlignment(Qt.AlignRight)
        self.lbl_mirror_fps = QLabel()
        self.lbl_mirror_fps.setAlignment(Qt.AlignRight)
        self.lbl_ui_fps = QLabel()
        self.lbl_ui_fps.setAlignment(Qt.AlignRight)
        
        column_2.addWidget(self.pb_flatten)
        column_2.addWidget(self.cb_closed)
        column_2.addLayout(f_layout)
        column_2.addLayout(bg_layout)
        column_2.addLayout(poke_layout)
        column_2.addWidget(self.cb_draw_boxes)
        column_2.addWidget(self.cb_draw_lines)
        column_2.addWidget(self.pb_quit)
        
        column_2.addWidget(self.lbl_error)
        column_2.addWidget(self.lbl_tip)
        column_2.addWidget(self.lbl_tilt)
        column_2.addWidget(self.lbl_cond)
        column_2.addWidget(self.lbl_sensor_fps)
        column_2.addWidget(self.lbl_mirror_fps)
        column_2.addWidget(self.lbl_ui_fps)
        
        column_2.addWidget(self.pb_poke)
        column_2.addWidget(self.pb_record_reference)
        column_2.addWidget(self.cb_logging)
        
        layout.addLayout(column_2,0,6,3,1)
        
        self.setLayout(layout)
        

    def flatten(self):
        self.mirror_mutex.lock()
        self.loop.mirror.flatten()
        self.mirror_mutex.unlock()

        
    @pyqtSlot()
    def update(self):
        self.mirror_mutex.lock()
        self.sensor_mutex.lock()
        sensor = self.loop.sensor
        mirror = self.loop.mirror

        temp = [(x,xerr,y,yerr) for x,xerr,y,yerr in
                zip(sensor.search_boxes.x,sensor.x_centroids,
                    sensor.search_boxes.y,sensor.y_centroids)]

        self.overlay_slopes.coords = []
        for x,xerr,y,yerr in temp:
            dx = (xerr-x)*ccfg.slope_line_magnification
            dy = (yerr-y)*ccfg.slope_line_magnification
            x2 = x+dx
            y2 = y+dy
            self.overlay_slopes.coords.append((x,x2,y,y2))

        self.id_spots.show(sensor.image)

        mirror_map = np.zeros(mirror.mirror_mask.shape)
        mirror_map[np.where(mirror.mirror_mask)] = mirror.get_command()[:]
        self.id_mirror.show(mirror_map)
        self.id_wavefront.show(sensor.wavefront)

        self.id_zoomed_spots.show(self.id_spots.zoomed())
        
        self.lbl_error.setText(ccfg.wavefront_error_fmt%(sensor.error*1e9))
        self.lbl_tip.setText(ccfg.tip_fmt%(sensor.tip*1000000))
        self.lbl_tilt.setText(ccfg.tilt_fmt%(sensor.tilt*1000000))
        self.lbl_cond.setText(ccfg.cond_fmt%(self.loop.get_condition_number()))
        self.lbl_sensor_fps.setText(ccfg.sensor_fps_fmt%sensor.frame_timer.fps)
        self.lbl_mirror_fps.setText(ccfg.mirror_fps_fmt%mirror.frame_timer.fps)
        self.lbl_ui_fps.setText(ccfg.ui_fps_fmt%self.frame_timer.fps)


        self.mirror_mutex.unlock()
        self.sensor_mutex.unlock()
            
    def select_single_spot(self,click):
        x = click.x()*self.downsample
        y = click.y()*self.downsample
        self.single_spot_index = self.loop.sensor.search_boxes.get_lenslet_index(x,y)

    def paintEvent(self,event):
        self.frame_timer.tick()




class ImageDisplay(QWidget):
    def __init__(self,name,downsample=None,clim=None,colormap=None,mouse_event_handler=None,image_min=None,image_max=None,width=512,height=512,zoom_height=ccfg.zoom_height,zoom_width=ccfg.zoom_width,zoomable=False,draw_boxes=False,draw_lines=False):
        
        super(ImageDisplay,self).__init__()
        self.name = name
        self.autoscale = False
        
        if downsample is None:
            self.downsample = ccfg.image_downsample_factor
        else:
            self.downsample = downsample
            
        self.sx = width
        self.sy = height
        self.draw_boxes = draw_boxes
        self.draw_lines = draw_lines
        self.zoomable = zoomable
        
        if clim is None:
            try:
                clim = np.loadtxt('.gui_settings/clim_%s.txt'%name)
            except Exception as e:
                self.autoscale = True
        
        self.clim = clim
        self.pixmap = QPixmap()
        self.label = QLabel()

        self.label.setScaledContents(False)

        
        
        self.image_max = image_max
        self.image_min = image_min
        self.zoom_width = zoom_width
        self.zoom_height = zoom_height
        
        layout = QHBoxLayout()
        layout.addWidget(self.label)

        if image_min is not None and image_max is not None and not self.autoscale:
            self.n_steps = 100
        
            self.cmin_slider = QSlider(Qt.Vertical)
            self.cmax_slider = QSlider(Qt.Vertical)

            self.cmin_slider.setMinimum(0)
            self.cmax_slider.setMinimum(0)

            self.cmin_slider.setSingleStep(1)
            self.cmax_slider.setSingleStep(1)

            self.cmin_slider.setPageStep(10)
            self.cmax_slider.setPageStep(10)

            self.cmin_slider.setMaximum(self.n_steps)
            self.cmax_slider.setMaximum(self.n_steps)

            self.cmin_slider.setValue(self.real2slider(self.clim[0]))
            self.cmax_slider.setValue(self.real2slider(self.clim[1]))

            self.cmin_slider.valueChanged.connect(self.set_cmin)
            self.cmax_slider.valueChanged.connect(self.set_cmax)
            
            layout.addWidget(self.cmin_slider)
            layout.addWidget(self.cmax_slider)

        
        self.setLayout(layout)

        self.zoomed = False
        self.colormap = colormap
        if self.colormap is not None:
            self.colortable = colortable(self.colormap)
        if mouse_event_handler is not None:
            self.mousePressEvent = mouse_event_handler
        else:
            self.mousePressEvent = self.zoom
            
        self.downsample = downsample
        
        data = np.random.rand(100,100)
        self.show(data)
        
        self.zoom_x1 = 0
        self.zoom_x2 = self.sx-1
        self.zoom_y1 = 0
        self.zoom_y2 = self.sy-1
        self.label_xoffset = 0.0
        self.label_yoffset = 0.0
        self.label_offsets_set = False
        
    def real2slider(self,val):
        # convert a real value into a slider value
        return round(int((val-float(self.image_min))/float(self.image_max-self.image_min)*self.n_steps))

    def slider2real(self,val):
        # convert a slider integer into a real value
        return float(val)/float(self.n_steps)*(self.image_max-self.image_min)+self.image_min
    
    def set_cmax(self,slider_value):
        self.clim = (self.clim[0],self.slider2real(slider_value))
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.clim)

    def set_cmin(self,slider_value):
        self.clim = (self.slider2real(slider_value),self.clim[1])
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.clim)
        
    def show(self,data,boxes=None,lines=None,mask=None):

        if mask is None:
            if boxes is not None:
                mask = np.ones(boxes[0].shape)
            elif lines is not None:
                mask = np.ones(lines[0].shape)
            else:
                assert (boxes is None) and (mask is None)
        
#        if self.name=='mirror':
#            print data[6,6]
            
        if self.autoscale:
            clim = (data.min(),data.max())
        else:
            clim = self.clim

        cmin,cmax = clim
        downsample = self.downsample
        data = data[::downsample,::downsample]
        
        if self.zoomed:
            x_scale = float(data.shape[1])/float(self.sx)
            y_scale = float(data.shape[0])/float(self.sy)

            zy1 = int(round(self.zoom_y1*y_scale))
            zy2 = int(round(self.zoom_y2*y_scale))
            zx1 = int(round(self.zoom_x1*x_scale))
            zx2 = int(round(self.zoom_x2*x_scale))
            
            #data = data[self.zoom_y1:self.zoom_y2,self.zoom_x1:self.zoom_x2]
            data = data[zy1:zy2,zx1:zx2]
            
        bmp = np.round(np.clip((data.astype(np.float)-cmin)/(cmax-cmin),0,1)*255).astype(np.uint8)
        sy,sx = bmp.shape
        n_bytes = bmp.nbytes
        bytes_per_line = int(n_bytes/sy)
        image = QImage(bmp,sy,sx,bytes_per_line,QImage.Format_Indexed8)
        if self.colormap is not None:
            image.setColorTable(self.colortable)
        self.pixmap.convertFromImage(image)


        self.label_xoffset = float(self.label.width()-self.pixmap.width())/2.0
        
        if boxes is not None and self.draw_boxes:
            x1vec,x2vec,y1vec,y2vec = boxes
            pen = QPen()
            pen.setColor(QColor(*ccfg.active_search_box_color))
            pen.setWidth(ccfg.search_box_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                if mask[index]:
                    width = float(x2 - x1 + 1)/float(self.downsample)
                    painter.drawRect(x1/float(self.downsample)-self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,width,width)
            painter.end()
            
        if lines is not None and self.draw_lines:
            x1vec,x2vec,y1vec,y2vec = lines
            pen = QPen()
            pen.setColor(QColor(*ccfg.slope_line_color))
            pen.setWidth(ccfg.slope_line_thickness)
            painter = QPainter()
            painter.begin(self.pixmap)
            painter.setPen(pen)
            for index,(x1,y1,x2,y2) in enumerate(zip(x1vec,y1vec,x2vec,y2vec)):
                if mask[index]:
                    painter.drawLine(QLine(x1/float(self.downsample)- self.zoom_x1,y1/float(self.downsample)-self.zoom_y1,x2/float(self.downsample)- self.zoom_x1,y2/float(self.downsample)- self.zoom_y1))
            painter.end()


        if sy==self.sy and sx==self.sx:
            self.label.setPixmap(self.pixmap)
        else:
            self.label.setPixmap(self.pixmap.scaled(self.sy,self.sx))
        
    def set_clim(self,clim):
        self.clim = clim

    def zoom(self,event):
        if not self.zoomable:
            return
        
        if self.zoom_width>=self.sx or self.zoom_height>=self.sy:
            return
        
        x,y = event.x(),event.y()
        if self.zoomed:
            self.zoomed = False
            self.zoom_x1 = 0
            self.zoom_x2 = self.sx-1
            self.zoom_y1 = 0
            self.zoom_y2 = self.sy-1
        else:
            self.zoomed = True
            self.zoom_x1 = x-self.zoom_width//2
            self.zoom_x2 = x+self.zoom_width//2
            self.zoom_y1 = y-self.zoom_height//2
            self.zoom_y2 = y+self.zoom_height//2
            if self.zoom_x1<0:
                dx = -self.zoom_x1
                self.zoom_x1+=dx
                self.zoom_x2+=dx
            if self.zoom_x2>self.sx-1:
                dx = self.zoom_x2-(self.sx-1)
                self.zoom_x1-=dx
                self.zoom_x2-=dx
            if self.zoom_y1<0:
                dy = -self.zoom_y1
                self.zoom_y1+=dy
                self.zoom_y2+=dy
            if self.zoom_y2>self.sy-1:
                dy = self.zoom_y2-(self.sy-1)
                self.zoom_y1-=dy
                self.zoom_y2-=dy

            if self.name=='spots':
                print self.label.width()
                print self.pixmap.width()
                print self.label_xoffset
                print 'zooming to %d,%d,%d,%d'%(self.zoom_x1,self.zoom_x2,self.zoom_y1,self.zoom_y2)

    def set_draw_lines(self,val):
        self.draw_lines = val

    def set_draw_boxes(self,val):
        self.draw_boxes = val
        
