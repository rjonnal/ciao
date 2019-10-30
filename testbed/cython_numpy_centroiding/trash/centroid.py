import numpy as np
from kungpao.config import kungpao_config as kcfg
from matplotlib import pyplot as plt
import glob

spots_images = glob.glob('/home/rjonnal/code/kungpao/data/spots/spots*.npy')
im = np.load(spots_images[0])
plt.imshow(im)
plt.show()
def get_spots_images():
    return spots_images

def build_sensor_searchbox_mask():
    ref = kcfg.REFERENCE_COORDINATES
    refx_vec = ref[:,0]
    refy_vec = ref[:,1]
    sb_width = kcfg.SEARCH_BOX_WIDTH_PX
    sy,sx = kcfg.SENSOR_HEIGHT_PX,kcfg.SENSOR_WIDTH_PX
    mask = np.ones((sy,sx))*(-1)
    n_ref = len(refx_vec)
    for ref_index,(refx,refy) in enumerate(zip(refx_vec,refy_vec)):
        print refx,refy


if __name__=='__main__':
    build_sensor_searchbox_mask()

