int test(int from, int to)
{
  int i;
  int s = 0;
 
  for (i = from; i < to; i++)
    if (i % 3 == 0)
      s += i;

  return s;
}

// This function takes the following arguments:
// Inputs:
//   short * image: a pointer to an image
//   short x: first column of region
//   short y: first row of region
//   short width: width of region
//   short height: height of region
// Outputs:
//   float * x_center_of_mass: a place to store the x COM
//   float * y_center_of_mass: a place to store the y COM
//   short * region_max: a pointer to a short to store the max pixel
//   short * region_min: a pointer to a short to store the min pixel
void centroid_region(short * image,
                     short image_width,
                     short image_height,
                     short x,
                     short region_width,
                     short y,
                     short region_height,
                     float * x_center_of_mass,
                     float * y_center_of_mass,
                     short * region_max,
                     short * region_min){
  short x_index;
  short y_index;

  float numerator = 0.0;
  float denominator = 0.0;

  float pixel = 0.0;
  
  for (x_index=x; x_index<x+region_width; x++){
    for (y_index=y; y_index<y+region_height; y++){
      pixel = image[y_index * region_width + x_index];
      
    }
  }

}
                     
