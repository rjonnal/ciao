/*************************************************************************************/
/*
 * File name: MDigProcess.c
 *
 * Synopsis:  This program shows the use of the MdigProcess() function to do perform
 *			  real-time processing.
 *
 *            The user's processing code is written in a hook function that
 *            will be called for each frame grabbed (see ProcessingFunction()).
 *
 *      Note: The average processing time must be shorter than the grab time or
 *            some frames will be missed. Also, if the processing results are not
 *            displayed and the frame count is not drawn or printed, the
 *            CPU usage is reduced significantly.
 */
#include <mil.h>
#include <conio.h>
#include <stdlib.h>

 /* Number of images in the buffering grab queue.
    Generally, increasing this number gives better real-time grab.
  */
#define BUFFERING_SIZE_MAX 20

/* User's processing function prototype. */
long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr);

/* User's processing function hook data structure. */
typedef struct
   {
   MIL_ID  MilImageDisp;
   long    ProcessedImageCount;
   } HookDataStruct;


/* Main function. */
/* ---------------*/

void main(void)
{
   MIL_ID MilApplication;
   MIL_ID MilSystem     ;
   MIL_ID MilDigitizer  ;
   MIL_ID MilDisplay    ;
   MIL_ID MilImageDisp  ;
   MIL_ID MilGrabBufferList[BUFFERING_SIZE_MAX] = { 0 };
   long   MilGrabBufferListSize;
   long   ProcessFrameCount  = 0;
   long   NbFrames           = 0;
   double ProcessFrameRate   = 0;
   HookDataStruct UserHookData;

   /* Allocate defaults. */
   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, &MilDisplay,
                                              &MilDigitizer, &MilImageDisp);

   /* Allocate the grab buffers and clear them. */
   MappControl(M_ERROR, M_PRINT_DISABLE);
   for(MilGrabBufferListSize = 0; MilGrabBufferListSize<BUFFERING_SIZE_MAX; MilGrabBufferListSize++)
      {
      MbufAlloc2d(MilSystem,
                  MdigInquire(MilDigitizer, M_SIZE_X, M_NULL),
                  MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL),
                  M_DEF_IMAGE_TYPE,
                  M_IMAGE+M_GRAB+M_PROC,
                  &MilGrabBufferList[MilGrabBufferListSize]);

      if (MilGrabBufferList[MilGrabBufferListSize])
         {
         MbufClear(MilGrabBufferList[MilGrabBufferListSize], 0xFF);
         }
      else
         break;
      }
   MappControl(M_ERROR, M_PRINT_ENABLE);

   /* Free a buffer to leave space for possible temporary buffer. */
   MilGrabBufferListSize--;
   MbufFree(MilGrabBufferList[MilGrabBufferListSize]);

   /* Print a message. */
   printf("\nMULTIPLE BUFFERED PROCESSING.\n");
   printf("-----------------------------\n\n");
   printf("Press <Enter> to start.\n\n");

   /* Grab continuously on the display and wait for a key press. */
   MdigGrabContinuous(MilDigitizer, MilImageDisp);
   getch();

   /* Halt continuous grab. */
   MdigHalt(MilDigitizer);

   /* Initialize the User's processing function data structure. */
   UserHookData.MilImageDisp        = MilImageDisp;
   UserHookData.ProcessedImageCount = 0;

   /* Start the processing. The processing function is called for every frame grabbed. */
   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
                             M_START, M_DEFAULT, ProcessingFunction, &UserHookData);


   /* NOTE: Now the main() is free to perform other tasks while the processing is executing. */
   /* --------------------------------------------------------------------------------- */


   /* Print a message and wait for a key press after a minimum number of frames. */
   printf("Press <Enter> to stop.\n\n");
   getch();

   /* Stop the processing. */
   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
               M_STOP, M_DEFAULT, ProcessingFunction, &UserHookData);


   /* Print statistics. */
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_COUNT,  &ProcessFrameCount);
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_RATE,   &ProcessFrameRate);
   printf("\n\n%ld frames grabbed at %.1f frames/sec (%.1f ms/frame).\n",
                          ProcessFrameCount, ProcessFrameRate, 1000.0/ProcessFrameRate);
   printf("Press <Enter> to end.\n\n");
   getch();

   /* Free the grab buffers. */
   while(MilGrabBufferListSize > 0)
      MbufFree(MilGrabBufferList[--MilGrabBufferListSize]);

   /* Release defaults. */
   MappFreeDefault(MilApplication, MilSystem, MilDisplay, MilDigitizer, MilImageDisp);
}


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
   printf("Processing frame #%d.\r", UserHookDataPtr->ProcessedImageCount);
   MOs_ltoa(UserHookDataPtr->ProcessedImageCount, Text, 10);
   MgraText(M_DEFAULT, ModifiedBufferId, STRING_POS_X, STRING_POS_Y, Text);

   /* Perform the processing and update the display. */
   #if (!M_MIL_LITE)
      MimArith(ModifiedBufferId, M_NULL, UserHookDataPtr->MilImageDisp, M_NOT);
   #else
      MbufCopy(ModifiedBufferId, UserHookDataPtr->MilImageDisp);
   #endif

   return 0;
   }
