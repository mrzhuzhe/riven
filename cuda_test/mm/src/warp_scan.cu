#include <stdio.h>

__global__ void warpReduce(int* in) {
    int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    //int value = 31 - laneId;
    int value = in[threadIdx.x];
    int res = 0;
    // Use XOR mode to perform butterfly reduction
    unsigned int mask = __activemask();
    for (int i=16; i>=1; i/=2)
        //value += __shfl_xor_sync(0xffffffff, value, i, 32);
        //value += __shfl_down_sync(0xffffffff, value, i);
        value += __shfl_down_sync(mask, value, i);

    // "value" now contains the sum across all threads
    printf("LanId %d Thread %d value = %d res value = %d\n", laneId, threadIdx.x, value, res);
}

__global__ void scan4() {
    int laneId = threadIdx.x & 0x1f;
    // Seed sample starting value (inverse of lane ID)
    int value = 31 - laneId;

    // Loop to accumulate scan within my partition.
    // Scan requires log2(n) == 3 steps for 8 threads
    // It works by an accumulated sum up the warp
    // by 1, 2, 4, 8 etc. steps.
    for (int i=1; i<=4; i*=2) {
        // We do the __shfl_sync unconditionally so that we
        // can read even from threads which won't do a
        // sum, and then conditionally assign the result.
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }

    printf("Thread %d final value = %d\n", threadIdx.x, value);
}

int main() {
    //scan4<<< 1, 32 >>>();
    //cudaDeviceSynchronize();
    int n = 32;
    int *A;
    int *d_A;
    int _size = n * sizeof(int);
    A = (int *)malloc(_size);
    for (int i = 0; i < n; i++){
        A[i] = i;
    }
    cudaMalloc((void **)&d_A, _size);
    cudaMemcpy(d_A, A, _size, cudaMemcpyHostToDevice);

    printf("\n");
    warpReduce<<< 1, 32 >>>(d_A);
    cudaDeviceSynchronize();

    return 0;
}