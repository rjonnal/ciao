# CIAO: community-inspired adaptive optics
Python tools for controlling, simulating, and characterizing adaptive optics (AO) systems

# Setup and installation

## Prerequisites

These prerequisites assume you are using the default hardware (Alpao mirror and a SHWS based on a Basler Ace USB3 camera).

1. Install [Notepad++](https://notepad-plus-plus.org/download) or another editor.
2. Install [Git](https://git-scm.com/download/win)
3. Install [Anaconda for Python 2.7](https://www.anaconda.com/distribution/#download-section)
4. Install Alpao drivers
5. Clone this repository
6. Install the [Visual C++ compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266)
7. Install [Basler Pylon 5.2](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/pylon-5-2-0-windows/)
8. Install [pypylon](https://github.com/basler/pypylon/releases/download/1.4.0/pypylon-1.4.0-cp27-cp27m-win_amd64.whl). First download, then use 'pip install pypylon...amd64.whl'.

## Environment variable

Optionally, set an environment variable ```CIAO_ROOT`` to this repository's location, including the ```ciao```, e.g. ```C:\\programs\\ciao```.

# Creating mask files

The reference mask is a two-dimensional arrays of zeros and ones which specifies which of the Shack-Hartmann lenslets to use. The mirror mask specifies, similarly, the logical locations of active mirror actuators. 

The program ```ciao/calibration/make_mask.py``` can be used to generate both mask files. Premade masks can be found in ```ciao/calibration/example_masks/```.

After navigating into the ```calibration``` directory, the program is run at the command line using ```python make_mask.py N rad mask_filename.txt```, where ```N``` specifies the size of the (square) mask, ```rad``` specifies the radius of the inscribed active area, and ```mask_filename.txt``` is the file in which to store the output.

Masks for the mirror and Shack-Hartmann lenslet array must be created, and copied into the ```etc/dm``` and ```etc/ref``` directories.

# Creating a reference coordinate file

The reference coordinates are the (x,y) locations on the sensor, in pixel units, where spots are expected to fall when a planar wavefront is incident on the sensor. These are the coordinates with which the boundaries of the search boxes are defined, and the coordinates with which the local slope of the wavefront is computed, used to drive the mirror in closed loop or to reconstruct the wavefront and measure wavefront error.

The coordinates are stored in a file named, e.g. ```reference_coordinates.txt``` in the ```etc/ref/``` directory. This should be a comma-delimited plain text file, with N rows and two items per row, where N is the number of active lenslets (see **masks** above) and the two items are x and y coordinates, respectively.

Several approaches have been used to generate these coordinates, but a common approach is to shine a collimated beam on the sensor and record the positions (centers of mass) of the resulting spots. There is a bit of a catch-22 in this approach, however, since the definition of search boxes and calculation of centers of mass require coordinates to get started. A script, ```calibration/record_reference_coordinates.py``` is included to generate in initial set of coordinates, which can be used to bootstrap a more accurate set. It works by using the geometric centers of the lenslets to generate a fake spots image, and then cross-correlating it with a real image from the sensor. The process proceeds as follows:

1. Run ```python record_reference_coordinates.py N temp.txt```, where ```N``` is the number of sensor images to average.

2. Move ```temp.txt``` into the ```etc/ref``` directory and define ```reference_coordinates_filename``` in ```config.py``` accordingly.

3. Run CIAO and verify that the spots are roughly centered in the search boxes.

4. Click **```Record reference```**. This may need to be done more than once, because background noise in a search box that's not centered at the spot's center of mass causes a bias toward the reference coordinates, i.e. an underestimate of error. The residual wavefront error RMS may be used to verify that the coordinates have been recorded correctly, since apparent error due to shot noise will eventually reach a stable minimum. Typically this value should be $\leq 10\;nm$.

# Design principles

0. **Balance exploratory/educational goals with real-time goals**. This software is intended to be neither the highest-performance AO software nor the most [literate](https://en.wikipedia.org/wiki/Literate_programming), but to balance both goals. This is hard to achieve, and requires judgement calls, but some examples:

    a. We want the mirror object to be useful in real-time operation, running in its own thread, but also to be instantiable in [REPL](https://en.wikipedia.org/wiki/Read-eval-print-loop). Therefore, even though it may be faster and simpler to subclass a ```threading.Thread``` or ```QThread```, instead create threads and move the objects to threads as needed only in those classes used in the real-time case, such as ```ciao.Loop```. The same goes for mutexes; if employed, lock and unlock them from the real-time class, and don't employ them in classes meant to be used in both contexts.
    
    b. Qt signals and slots are used for event-driven programming, necessary for real-time operation. As with threads, above, instead of subclassing Qt signals and slots, use Qt decorators. This way, the functions can be called in REPL without trouble, but can also call each other via events in the real-time case.

1. **Use open source tools whenever possible**. The only exceptions should be closed-source drivers for AO components such as cameras and deformable mirrors.

2. **Use human-readable configuration, initialization, and log files.** Whenever possible, use plain text files, along with delimiters. Configuration and initialization files should be written in Python whenever possible.

3. **Avoid overspecification.** Specify parameters of the system in only one place. For example, since the number of lenslets is specified by the SHWS mask, don't specify it elsewhere. The size of the poke matrix, for instance, is implied by the SHWS and mirror masks, and needn't be specified anywhere else.

4. **Variable naming.** Class names should be one word with a capitalized first letter; if multiple words are necessary, use CamelCase. Variable names should be descriptive, with underscores between words. In order to maximize the exploratory and educational value of the code, when a variable has a unit, please append it to the end of the variable name, e.g. ```wavelength_m = 800e-9```.

# Topics for conversation

1. Other than condition number, what algorithmic or numerical tests can be employed to predict the performance of a given poke/control matrix?

2. What rules of thumb should be employed when deciding whether a spot should be used? Put differently, how should one choose ```rad``` when generating a SHWS mask, as described above?

