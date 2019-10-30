from PySide.QtCore import *
from PySide.QtGui import *
import cv2
import sys
from time import time

class MainApp(QWidget):
 
    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(320, 240)
        self.setup_ui()
        self.setup_camera()
        self.fps = 0
        self.t0 = time()
        self.N = 0.0
        
    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.fps_label = QLabel()
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
 
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.fps_label)
        self.main_layout.addWidget(self.quit_button)
 
        self.setLayout(self.main_layout)
 
    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.video_size.height())
 
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)
 
    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.t = time()-self.t0
        self.fps = self.N/self.t
        self.N = self.N+1.0
        
        self.fps_label.setText('%d,%0.3f,%0.3f'%(self.N,self.t,self.fps))
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())

sys.exit()

import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

class Camera:

    def __init__(self,exposure_time = 0.1):
        self.exposure_time = exposure_time



