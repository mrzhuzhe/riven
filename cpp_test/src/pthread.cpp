#include <iostream>
#include <pthread.h>

void* func1(void* args){
    printf("this is async 0 \n");
}

void* func2(void* args){
    printf("this is async 1 \n");
}

int main(){
    printf("pthread test go\n");
    pthread_t ptid;
    pthread_t ptid1;

    pthread_create(&ptid, NULL, &func1, NULL);
    pthread_create(&ptid1, NULL, &func2, NULL);
    printf("sync \n");

    pthread_join(ptid, NULL);
    printf("join \n");
    //pthread_exit(NULL);
    return 0;
} 