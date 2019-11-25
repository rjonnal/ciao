from PyQt5.QtMultimedia import QSound,QSoundEffect,QAudioDeviceInfo
from PyQt5.QtCore import QUrl
import ciao_config as ccfg
import sys
import numpy as np
import time
import pygame

class Beeper:

    def __init__(self):

        self.active = ('audio_directory' in dir(ccfg) and 'error_tones' in dir(ccfg))

        if self.active:
            self.tone_dict = {}
            for minmax,tonefn in ccfg.error_tones:
                key = self.err_to_int(minmax[0])
                val = QSoundEffect()
                val.setSource(QUrl(tonefn))

                self.tone_dict[key] = val
                #self.tonepg_dict[key] = pygame.mixer.Sound(tonefn)
                
    def err_to_int(self,err):
        return int(np.floor(err*1e8))

    def beep(self,error_in_nm):
        if self.active:
            k = self.err_to_int(error_in_nm)
            if k in self.tone_dict.keys():
                se = self.tone_dict[k]
                se.play()
                print 'play %0.1f'%(error_in_nm*1e9)

    
    def beep1(self,error_in_nm):
        if self.active:
            k = self.err_to_int(error_in_nm)
            if k in self.tonepg_dict.keys():
                pygame.mixer.Sound.play(self.tonepg_dict[k])
                
