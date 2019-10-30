#include "kungpao_camera.h"

MIL_ID MilGrabBufferList[BUFFERING_SIZE_MAX] = { 0 };
long   MilGrabBufferListSize;
long   ProcessFrameCount  = 0;
double ProcessFrameRate   = 0;

/* User's processing function hook data structure. */
typedef struct
   {
   MIL_ID  MilImageDisp;
   long    ProcessedImageCount;
   } HookDataStruct;

typedef struct
{
    float centroid_x;
    float centroid_y;
    unsigned short dc;
    unsigned short box_max;
    unsigned short box_min;
    float box_total;
} search_box;
   
   
HookDataStruct UserHookData;
   

void setup_default(void){
   /* Allocate defaults. */
//   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, &MilDisplay,
//                                              &MilDigitizer, &MilImageDisp);
   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, M_NULL,
                                              &MilDigitizer, &MilImageDisp);
}


void setup(MIL_CONST_TEXT_PTR system_name, MIL_CONST_TEXT_PTR camera_filename){
    printf("kungpao_camera setup system_name: ");
    printf(system_name);
    printf("\n");

    printf("kungpao_camera setup camera_filename: ");
    printf(camera_filename);
    printf("\n");
    
    MappAlloc(M_DEFAULT,&MilApplication);
    MsysAlloc(MIL_TEXT(system_name),M_DEFAULT,M_DEFAULT,&MilSystem);

    MdigAlloc(MilSystem,M_DEFAULT,MIL_TEXT(camera_filename),M_DEFAULT,&MilDigitizer);

    long size_x = MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
    printf("MdigInquire M_SIZE_X result: %d.\n",size_x);
    long size_y = MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
    printf("MdigInquire M_SIZE_Y result: %d.\n",size_y);
    
    MbufAlloc2d(MilSystem,size_x,size_y,16,M_IMAGE+M_GRAB,&MilImageDisp);
}

void start(void)
{
   /* Allocate the grab buffers and clear them. */
   MappControl(M_ERROR, M_PRINT_DISABLE);
   long size_x = MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
   long size_y = MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
   printf("Image dimensions: %d size_x %d.\n",size_x,size_y);
   for(MilGrabBufferListSize = 0; MilGrabBufferListSize<BUFFERING_SIZE_MAX; MilGrabBufferListSize++)
      {
      MbufAlloc2d(MilSystem,
                  size_x,
                  size_y,
                  16,
                  M_IMAGE+M_GRAB+M_PROC,
                  &MilGrabBufferList[MilGrabBufferListSize]);
      printf("Allocated buffer: %d\n",MilGrabBufferList[MilGrabBufferListSize]);
      if (MilGrabBufferList[MilGrabBufferListSize])
         {
         MbufClear(MilGrabBufferList[MilGrabBufferListSize], 0xFF);
         }
      else
         break;
      }
   MappControl(M_ERROR, M_PRINT_ENABLE);

   
   UserHookData.MilImageDisp        = MilImageDisp;
   UserHookData.ProcessedImageCount = 0;

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
                             M_START, M_DEFAULT, ProcessingFunction, &UserHookData);

}

void stop(void){

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
               M_STOP, M_DEFAULT, ProcessingFunction, &UserHookData);


   /* Print statistics. */
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_COUNT,  &ProcessFrameCount);
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_RATE,   &ProcessFrameRate);
   printf("\n\n%ld frames grabbed at %.1f frames/sec (%.1f ms/frame).\n",
                          ProcessFrameCount, ProcessFrameRate, 1000.0/ProcessFrameRate);
}

void release(void){
   /* Free the grab buffers. */
   while(MilGrabBufferListSize > 0)
      MbufFree(MilGrabBufferList[--MilGrabBufferListSize]);

   /* Release defaults. */
   MappFreeDefault(MilApplication, MilSystem, MilDisplay, MilDigitizer, MilImageDisp);
}


/* -----------------------------------------------------------------------*/
/* User's processing function called every time a grab buffer is modified. */
/* -----------------------------------------------------------------------*/

/* Local defines. */
#define STRING_LENGTH_MAX  20
#define STRING_POS_X       20
#define STRING_POS_Y       20

long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr)
    {
    HookDataStruct *UserHookDataPtr = (HookDataStruct *)HookDataPtr;
    MIL_ID ModifiedBufferId;
    MIL_TEXT_CHAR Text[STRING_LENGTH_MAX]= {'\0',};

    /* Retrieve the MIL_ID of the grabbed buffer. */
    MdigGetHookInfo(HookId, M_MODIFIED_BUFFER+M_BUFFER_ID, &ModifiedBufferId);

    /* Print and draw the frame count. */
    UserHookDataPtr->ProcessedImageCount++;
    printf("Processing frame #%d %d.\r", UserHookDataPtr->ProcessedImageCount, UserHookDataPtr->MilImageDisp);

    MbufCopy(ModifiedBufferId, UserHookDataPtr->MilImageDisp);
   
    return 0;
    }
    
void get_current_image(void * data_pointer)
{
    MbufGet((&UserHookData)->MilImageDisp,data_pointer);
}

long get_size_x(void)
{
   return MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
}

long get_size_y(void)
{
   return MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
}

static unsigned int counter, xs, xe, ys, ye, xes, xee, yes, yee, x, y, yScaled, xScaled, widthScaled;
static float denom, xsum, ysum, scaledPixelFloat, lastImageRange, box_total;
static short image_max, image_min, box_max, box_min, edge_rad, edge_max;
static int pixel;
static unsigned char scaledPixelUChar;

#if SERIAL
void compute_centroid(unsigned short * image_in,
                      unsigned short * image_out,
                      float ref_x, 
                      float ref_y, 
                      unsigned short rad, 
                      search_box * sb, 
                      unsigned short index,
                      unsigned short width,
                      unsigned short height)
{
    xsum = 0.0;
    ysum = 0.0;
    denom = 0.0;
    box_max = 0;
    box_min = 2 ^ 15;
    box_total = 0.0;
    
    // boundaries fo the box
    xs = (int)round(ref_x) - rad;
    xe = (int)round(ref_x) + rad;
    ys = (int)round(ref_y) - rad;
    ye = (int)round(ref_y) + rad;
    
    edge_rad = 2;
    if (edge_rad>rad) edge_rad = rad;
    // boundaries of search regions near centers of each edge
    xes = (int)round(ref_x) - edge_rad;
    xee = (int)round(ref_x) + edge_rad;
    yes = (int)round(ref_y) - edge_rad;
    yee = (int)round(ref_y) + edge_rad;
    
    // survey box edge for brightest(ish) pixel, to use as DC
    edge_max = 0;
    for (x = xes; x <= xee; x = x + 1){
        pixel = (int)image_in[ys * width + x];
        //image_out[ys*width+x] = 500; // used to visualize box edges in testing
        if (pixel>edge_max) edge_max = pixel;
        pixel = (int)image_in[ye * width + x];
        //image_out[ye*width+x] = 500; // used to visualize box edges in testing
        if (pixel>edge_max) edge_max = pixel;
    }
    for (y = yes; y <= yee; y = y + 1){
        pixel = (int)image_in[y * width + xs];
        //image_out[y*width+xs] = 500; // used to visualize box edges in testing
        if (pixel>edge_max) edge_max = pixel;
        pixel = (int)image_in[y * width + xe];
        //image_out[y*width+xe] = 500; // used to visualize box edges in testing
        if (pixel>edge_max) edge_max = pixel;
    }
    sb->dc = edge_max;
    for (x = xs; x <= xe; x = x + 1){
        for (y = ys; y <= ye; y = y + 1){
            pixel = (int)image_in[y * width + x];
            //printf("pixel %d, box_max %d, box_min %d\n",pixel,box_max,box_min);
            if (pixel>box_max) box_max = pixel;
            if (pixel<box_min) box_min = pixel;

            pixel = pixel  - (int)edge_max;
            if (pixel<0) pixel=0;
            image_out[y * width + x] = pixel;
            xsum = xsum + x * pixel;
            ysum = ysum + y * pixel;
            denom = denom + pixel;
        }
    }
    sb->box_max = box_max;
    sb->box_min = box_min;
    sb->centroid_x = xsum / denom;
    sb->centroid_y = ysum / denom;
}
#endif

#if PARALLEL
void compute_centroid(unsigned short * image_in,
                      unsigned short * image_out,
                      float ref_x, 
                      float ref_y, 
                      unsigned short rad, 
                      search_box * centroid, 
                      unsigned short index)
{
    g;
}
#endif