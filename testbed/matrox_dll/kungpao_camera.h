#include <mil.h>
#include <conio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <math.h>

#define BUFFERING_SIZE_MAX 20
#define SERIAL 1
#define PARALLEL 1-SERIAL


MIL_ID MilApplication;
MIL_ID MilSystem     ;
MIL_ID MilDigitizer  ;
MIL_ID MilDisplay    ;
MIL_ID MilImageDisp  ;

/* User's processing function prototype. */
long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr);