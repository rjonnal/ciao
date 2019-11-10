#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <omp.h>

// Need to specify extern "C" (__declspec(dllexport) in Windows)
// in order to avoid name mangling by compiler, since we want to
// expose these symbols to Python cleanly.

#ifdef _WIN32
#define NO_MANGLE extern "C" __declspec(dllexport)
#elif __linux__
#define NO_MANGLE extern "C"
#endif

// Signature for centroiding function:
// inputs:
//    image: pointer to the int16/short Shack-Hartmann image
//    image_width: short containing number of pixels in each row
//    ref_x and ref_y: pointers to floating point reference coordinates
//    search_box_size: short specifying width and height of search boxes
//    n_search_boxes: length of ref_x, ref_y, x_output, and y_output
//    x_output and y_output: where to put the derivatives (distances
//        between centroids and references)

NO_MANGLE void centroid_spots_omp(short * image,
                              short image_width,
                              double * x_ref,
                              double * y_ref,
                              short search_box_size,
                              short n_search_boxes,
                              double * x_output,
                              double * y_output);

NO_MANGLE void centroid_spots(short * image,
                              short image_width,
                              double * x_ref,
                              double * y_ref,
                              short search_box_size,
                              short n_search_boxes,
                              double * x_output,
                              double * y_output);

