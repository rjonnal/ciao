#include "centroiding.h"

// variables to hold counters and start/end
// coordinates for x and y
static unsigned int x, y, x_1, x_2, y_1, y_2;

// variables to accumulate moment (numerators) and
// total energy (denominator)
static double x_numerator, y_numerator, denominator, pixel;
static short n;

int omp_get_thread_num();
int omp_get_num_threads();
  
void centroid_spots_omp(short * image,
                    short image_width,
                    double * x_ref,
                    double * y_ref,
                    short search_box_size,
                    short n_search_boxes,
                    double * x_output,
                    double * y_output){
  
  // iterate through the search boxes
  #pragma omp parallel num_threads(4)
  {
  #pragma omp for
  for (n = 0; n < n_search_boxes; n += 1){
    //printf("Search box %d computed in thread number %d\n",n,omp_get_thread_num());
    // floating point accumulators for coordinate*intensity
    // (x_ and y_numerator) and intensity (denominator)
    x_numerator = 0.0;
    y_numerator = 0.0;
    denominator = 0.0;

    // [x1,x2] and [y1,y2] are closed intervals for
    // computing center of mass (that is, x2 and y2
    // are included in the computation; use <= in
    // associated for loops
    // 
    x_1 = (unsigned int)round(x_ref[n]) - search_box_size/2; // truncate to int
    x_2 = (unsigned int)round(x_ref[n]) + search_box_size/2;
    y_1 = (unsigned int)round(y_ref[n]) - search_box_size/2;
    y_2 = (unsigned int)round(y_ref[n]) + search_box_size/2;

    for (x = x_1; x <= x_2; x +=1){
      for (y = y_1; y <= y_2; y+=1){
        pixel = (double)image[y * image_width + x];
        denominator += pixel;
        x_numerator += pixel*(double)x;
        y_numerator += pixel*(double)y;
      }
    }
    x_output[n] = x_numerator/denominator - x_ref[n];
    y_output[n] = y_numerator/denominator - y_ref[n];
  }
  }
}


void centroid_spots(short * image,
                    short image_width,
                    double * x_ref,
                    double * y_ref,
                    short search_box_size,
                    short n_search_boxes,
                    double * x_output,
                    double * y_output){
  
  // iterate through the search boxes
  for (n = 0; n < n_search_boxes; n += 1){
    //printf("Search box %d computed in thread number %d\n",n,omp_get_thread_num());
    // floating point accumulators for coordinate*intensity
    // (x_ and y_numerator) and intensity (denominator)
    x_numerator = 0.0;
    y_numerator = 0.0;
    denominator = 0.0;

    // [x1,x2] and [y1,y2] are closed intervals for
    // computing center of mass (that is, x2 and y2
    // are included in the computation; use <= in
    // associated for loops
    // 
    x_1 = (unsigned int)round(x_ref[n]) - search_box_size/2; // truncate to int
    x_2 = (unsigned int)round(x_ref[n]) + search_box_size/2;
    y_1 = (unsigned int)round(y_ref[n]) - search_box_size/2;
    y_2 = (unsigned int)round(y_ref[n]) + search_box_size/2;

    for (x = x_1; x <= x_2; x +=1){
      for (y = y_1; y <= y_2; y+=1){
        pixel = (double)image[y * image_width + x];
        denominator += pixel;
        x_numerator += pixel*(double)x;
        y_numerator += pixel*(double)y;
      }
    }
    x_output[n] = x_numerator/denominator - x_ref[n];
    y_output[n] = y_numerator/denominator - y_ref[n];
  }
}
