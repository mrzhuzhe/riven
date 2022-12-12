#include <stdio.h>
#include <stdlib.h>

#define N 512

void host_add(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        c[idx] = a[idx] + b[idx];
}

void fill_array(int *data) {
    for (int idx=0;idx<N;idx++)
        data[idx] = idx;
}

void print_output(int *a, int *b, int *c) {
    for (int idx=0;idx<N;idx++)
        printf("\n %d = %d + %d", c[idx], a[idx], b[idx]);
}

int main(void) {
    int *a, *b, *c;
    int size = N * sizeof(int);
    a = (int *)malloc(size); fill_array(a);
    b = (int *)malloc(size); fill_array(b);
    c = (int *)malloc(size);
    host_add(a, b, c);
    print_output(a, b, c);
    free(a);
    free(b);
    free(c);

    return 0;
}