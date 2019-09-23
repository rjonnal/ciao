# create a reference coord set based on an old one
# permitting shifts in x and y

import numpy as np
import os,sys
import kungpao.config as kcfg

if len(sys.argv)<4:
    print 'Usage: python shift_reference.py INPUT_FILE DX DY [OUTPUT_FILE]'
    print 'Reads input reference file, adds DX and DY (in units of lenslet'
    print 'count), and outputs to OUTPUT_FILE if given or standard output if not'
    sys.exit()

px = kcfg.pixel_size_m
pitch = kcfg.lenslet_pitch_m
c = pitch/px

in_file = sys.argv[1]
dx = float(sys.argv[2])
dy = float(sys.argv[3])

try:
    out_file = sys.argv[4]
except Exception as e:
    out_file = None

ref = np.loadtxt(in_file)
ref[:,0] = ref[:,0] + dx*c
ref[:,1] = ref[:,1] + dy*c

if out_file is None:
    for k in range(ref.shape[0]):
        print '%0.3f,%0.3f,'%(ref[k,0],ref[k,1])
else:
    np.savetxt(out_file,ref)

