#define _MULTI_THREADED
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
 
void foo(void);  /* Functions that use the threadSpecific data */
void bar(void);
void dataDestructor(void *data);
 
#define checkResults(string, val) {             \
 if (val) {                                     \
   printf("Failed with %d at %s", val, string); \
   exit(1);                                     \
 }                                              \
}
 
typedef struct {
  int          threadSpecific1;
  int          threadSpecific2;
} threadSpecific_data_t;
 
#define                 NUMTHREADS   2
pthread_key_t           threadSpecificKey;

unsigned int pthread_getthreadid_np(){
    return (1, 1);
}
 
void *theThread(void *parm)
{
   int               rc;
   threadSpecific_data_t    *gData;
   printf("Thread %.8x %.8x: Entered\n", pthread_getthreadid_np());
   gData = (threadSpecific_data_t *)parm;
   rc = pthread_setspecific(threadSpecificKey, gData);
   checkResults("pthread_setspecific()\n", rc);
   foo();

   threadSpecific_data_t *gData2 = (threadSpecific_data_t *)pthread_getspecific(threadSpecificKey);
   gData2->threadSpecific1 = 9999;
   gData2->threadSpecific2 = 9999;
   printf("Thread %.8x %.8x: 999(), threadSpecific data=%d %d\n",
          pthread_getthreadid_np(), gData2->threadSpecific1, gData2->threadSpecific2);


   return NULL;
}
 
void foo() {
   threadSpecific_data_t *gData = (threadSpecific_data_t *)pthread_getspecific(threadSpecificKey);
   printf("Thread %.8x %.8x: foo(), threadSpecific data=%d %d\n",
          pthread_getthreadid_np(), gData->threadSpecific1, gData->threadSpecific2);
   bar();
}
 
void bar() {
   threadSpecific_data_t *gData = (threadSpecific_data_t *)pthread_getspecific(threadSpecificKey);
   printf("Thread %.8x %.8x: bar(), threadSpecific data=%d %d\n",
          pthread_getthreadid_np(), gData->threadSpecific1, gData->threadSpecific2);
   gData->threadSpecific1 = 9999;
   return;
}
 
void dataDestructor(void *data) {
   printf("Thread %.8x %.8x: Free data\n", pthread_getthreadid_np());
   pthread_setspecific(threadSpecificKey, NULL);
   free(data);
}
 
 
int main(int argc, char **argv)
{
  pthread_t             thread[NUMTHREADS];
  int                   rc=0;
  int                   i;
  threadSpecific_data_t        *gData;
 
  printf("Enter Testcase - %s\n", argv[0]);
  rc = pthread_key_create(&threadSpecificKey, dataDestructor);
  checkResults("pthread_key_create()\n", rc);
 
  printf("Create/start threads\n");
  for (i=0; i <NUMTHREADS; i++) {
       /* Create per-thread threadSpecific data and pass it to the thread */
     gData = (threadSpecific_data_t *)malloc(sizeof(threadSpecific_data_t));
     gData->threadSpecific1 = i;
     gData->threadSpecific2 = (i+1)*2;
     rc = pthread_create(&thread[i], NULL, theThread, gData);
     checkResults("pthread_create()\n", rc);
  }
 
  printf("Wait for the threads to complete, and release their resources\n");
  for (i=0; i <NUMTHREADS; i++) {
  rc = pthread_join(thread[i], NULL);
     checkResults("pthread_join()\n", rc);
  }
   
  printf("  %d  \n", gData->threadSpecific2);

  pthread_key_delete(threadSpecificKey);
  printf("Main completed\n");
  return 0;
}