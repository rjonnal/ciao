#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

long foo;
long bar;
void sleep(int x);

// A normal C function that is executed as a thread 
// when its name is specified in pthread_create()
void *myThreadFun(void *vargp)
{
    sleep(1.5);
    printf("Printing GeeksQuiz from Thread \n");
    return NULL;
}



int main()
{
    pthread_t tid;
    printf("Before Thread\n");
    pthread_create(&tid, NULL, myThreadFun, NULL);
    pthread_join(tid, NULL);
    printf("After Thread\n");
    exit(0);
}
