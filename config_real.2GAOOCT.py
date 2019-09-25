# A set of name-value pairs specifying local configuration of
# ciao installation. Where appropriate, each parameter's final
# characters represent units. 

##############################################################
# A unique, permanent identifier for the optical system
# associated with this installation of ciao:
system_id = '2GAOOCT'

# An identifier for the deformable mirror, used to load
# the correct configuration files:
mirror_id = 'alpaoDM97-15-012'

image_width_px = 2048
image_height_px = 2048
bit_depth = 12

# the program will try to get this value from the environment
# variable CIAO_ROOT, but if it cannot it will default to this
ciao_root_default = 'c:/code/ciao'

# read ciao_path from the environment, if possible
# if not, use the default value specified above
try:
    import os
    ciao_root = os.environ['CIAO_PATH']
except Exception as e:
    print 'Could not find environment varialbe CIAO_PATH'
    print 'Using default value %s'%ciao_root_default
    ciao_root = ciao_root_default

# define some directories for configuration files
reference_directory = ciao_root + '/etc/ref/'
dm_directory = ciao_root + '/etc/dm/'
dcf_directory = ciao_root + '/etc/dcf/'
poke_directory = ciao_root + '/etc/ctrl/'
logging_directory = ciao_root + '/log'
simulator_cache_directory = '.simulator_cache'
simulated_camera_image_directory = ciao_root + '/data/spots/'


# The reference coordinates need a starting point (see README.md), and these
# were recorded using calibration/record_reference_coordinates.py:
reference_coordinates_bootstrap_filename = reference_directory + 'reference_initial.txt'
reference_coordinates_filename = reference_directory + 'reference.txt'
reference_mask_filename = reference_directory + 'reference_mask.txt'
poke_filename = poke_directory + 'poke.txt'

# sensor settings:
reference_n_measurements = 10
lenslet_pitch_m = 500e-6
lenslet_focal_length_m = 30.0e-3
pixel_size_m = 5.5e-6
beam_diameter_m = 10e-3
interface_scale_factor = 0.5
wavelength_m = 840e-9
estimate_background = True
background_correction = -100
search_box_half_width = 39 #9
spots_threshold = 100.0
sensor_update_rate = 20.0
sensor_filter_lenslets = False
sensor_reconstruct_wavefront = True
sensor_remove_tip_tilt = True
centroiding_num_threads = 1
iterative_centroiding_step = 2
centroiding_iterations = 1

mirror_update_rate = 200.0
mirror_n_actuators = 97
mirror_flat_filename = ciao_root + '/etc/dm/flat2G.txt'
mirror_mask_filename = ciao_root + '/etc/dm/mirror_mask.txt'
mirror_command_max = 1.0
mirror_command_min = -1.0
mirror_settling_time_s = 0.001

poke_command_max = 0.2
poke_command_min = -0.2
poke_n_command_steps = 5

ctrl_dictionary_max_size = 10

loop_n_control_modes = 50
loop_gain = 0.3
loop_loss = 0.01

n_zernike_terms = 66
zernike_dioptric_equivalent = 1.5


# UI settings:
image_downsample_factor = 1
active_search_box_color = (16,127,16,255)
inactive_search_box_color = (0,63,127,255)
search_box_thickness = 0.2
show_search_boxes = True
show_slope_lines = True
slope_line_thickness = 3.0
slope_line_color = (200,100,100,155)
slope_line_magnification = 10
spots_colormap = 'bone'
wavefront_colormap = 'jet'
wavefront_clim = (-1e-6,1e-6)
mirror_colormap = 'mirror'
mirror_clim = (-1,1)
zoom_width = 50
zoom_height = 50
single_spot_color = (255,63,63,255)
single_spot_thickness = 2.0



ui_fps_fmt = '%0.2f Hz (UI)'
sensor_fps_fmt = '%0.2f Hz (Sensor)'
mirror_fps_fmt = '%0.2f Hz (Mirror)'
wavefront_error_fmt = '%0.1f nm RMS (Error)'
tip_fmt = '%0.4f mrad (Tip)'
tilt_fmt = '%0.4f mrad (Tilt)'
cond_fmt = '%0.2f (Condition)'

search_box_half_width_max = int(lenslet_pitch_m/pixel_size_m)//2

rigorous_iteration = False
if rigorous_iteration:
    # First, calculate the PSF FWHM for the lenslets:
    import math
    lenslet_psf_fwhm_m = 1.22*wavelength_m*lenslet_focal_length_m/lenslet_pitch_m
    # Now see how many pixels this is:
    lenslet_psf_fwhm_px = lenslet_psf_fwhm_m/pixel_size_m 

    diffraction_limited_width_px = round(math.ceil(lenslet_psf_fwhm_px))
    if diffraction_limited_width_px%2==0:
        diffraction_limited_width_px+=1
    diffraction_limited_half_width_px = (diffraction_limited_width_px-1)//2

    iterative_centroiding_step = 1
    centroiding_iterations = int(round((search_box_half_width-diffraction_limited_half_width_px)//iterative_centroiding_step))

