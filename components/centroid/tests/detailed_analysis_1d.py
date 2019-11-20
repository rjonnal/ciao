from matplotlib import pyplot as plt
import numpy as np
import sys,os

sensor_size = 100
x = np.arange(sensor_size)
ref_x = x.mean()

def create_spot(sensor_size=15,x0=None,
                sigma=None,amplitude=1000.0,
                dc=100.0,round_output=False,noise_gain=1.0,do_plot=0.0):
    if x0 is None:
        x0 = (sensor_size-1)/2.0
    if sigma is None:
        sigma = sensor_size/100.0
    x = np.arange(sensor_size)
    light = amplitude*np.exp(-(x-x0)**2/(2*sigma**2))
    shot_noise = np.sqrt(light)*np.random.randn(len(light))*noise_gain
    read_noise = np.random.randn(len(light))*noise_gain
    noise = shot_noise+read_noise
    signal = light+noise+dc
    if round_output:
        signal = np.round(signal)
    if do_plot:
        plt.cla()
        plt.plot(signal)
        plt.pause(do_plot)
    return signal


def centroid(signal,x=None):
    if x is None:
        x = np.arange(len(signal))
    return np.sum(x*signal)/np.sum(signal)


def iterative_centroid(signal,ref_x,sb_half_width,n_iterations,iteration_step_px):
    x = np.arange(len(signal))
    ref_x_temp = ref_x
    for k in range(n_iterations):
        left = int(round(ref_x_temp-sb_half_width+k*iteration_step_px))
        right = int(round(ref_x_temp+sb_half_width-k*iteration_step_px))
        sig_temp = signal[left:right+1]
        x_temp = x[left:right+1]
        ref_x_temp = centroid(sig_temp,x_temp)
    return ref_x_temp


def maxcentered_centroid(signal,ref_x,sb_half_width,c_half_width):
    x = np.arange(len(signal))
    left = int(round(ref_x-sb_half_width))
    right = int(round(ref_x+sb_half_width))
    # copy to avoid modifying signal, since we'll be using the
    # same signal to test multiple algorithms
    # also, put zeros outside the search box so that we can
    # just use argmax on the whole vector
    temp = np.zeros(signal.shape)
    temp[left:right+1] = signal[left:right+1]
    x_peak = np.argmax(temp)

    # recenter the smaller centroiding box
    # on x_peak
    left = int(round(x_peak-c_half_width))
    right = int(round(x_peak+c_half_width))
    temp = temp[left:right+1]
    x = x[left:right+1]
    return centroid(temp,x)
    #return x_peak



def spot_and_com(x0,sigma,sb_half_width,n_iterations,iteration_step_px,c_half_width):
    sensor_size = 100
    amplitude = 1000.0
    dc = 100.0
    round_output = True
    noise_gain = 1.0
    do_plot=False
    x = np.arange(sensor_size)
    ref_x = x.mean()
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=dc,round_output=round_output,amplitude=amplitude,sigma=sigma)
    results = [None]*3
    results[0] = centroid(spot)
    results[1] = iterative_centroid(spot,ref_x,10,3,2) # this leads to a final size of FWHM
    results[2] = steve_centroid(spot,ref_x,10,c_half_width)
    return results

    
sigma = 3.0
fwhm = 2*np.sqrt(2*np.log(2))*sigma
c_half_width = np.ceil(fwhm)//2
noise_gain = 1.0
labels = ['simple COM','iterative COM','max-centered COM']

# test the impact of DC on the three methods
dc_range = np.arange(0,500)
results = np.zeros((len(dc_range),3))
for idx,dc in enumerate(dc_range):
    x0 = 45.0
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=dc,round_output=True,amplitude=1000.0,sigma=3.0)
    results[idx,0] = centroid(spot)
    results[idx,1] = iterative_centroid(spot,ref_x,10,3,2) # this leads to a final size of FWHM
    results[idx,2] = steve_centroid(spot,ref_x,10,c_half_width)

plt.plot(dc_range,results)
plt.legend(labels)

# test the impact of DC on the three methods
x0_range = np.arange(40.0,60.0,.05)
results = np.zeros((len(x0_range),3))
for idx,x0 in enumerate(x0_range):
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=100.0,round_output=True,amplitude=1000.0,sigma=3.0)
    results[idx,0] = centroid(spot)
    results[idx,1] = iterative_centroid(spot,ref_x,10,3,2) # this leads to a final size of FWHM
    results[idx,2] = steve_centroid(spot,ref_x,10,c_half_width)

plt.figure()
plt.plot(x0_range-ref_x,(results.T-x0_range).T)
plt.legend(labels)








plt.show()



sys.exit()
