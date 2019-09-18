# create a reference coord set based on an old one
# permitting shifts in x and y

import numpy as np
import os,sys

in_file = sys.argv[1]

ref = np.loadtxt(in_file)
print ref
