#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 128
#define SOFTENING 1e-9f

typedef struct {
    float4 *pos, *vel;
} NBodySystem;

void generateRandomizeBodies(float *data, int n){
    float max = (float)RAND_MAX;
        for  (int i =0; i<n; i++){
            data[i] = 2.0f * (rand()/max)-1.0f;
        }
}

__global__ void calculateBodyForce(float4 *p, float4 *v, float dt, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<n){
        float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
        for (int tile = 0; tile < gridDim.x ; tile++){
            __shared__ float3 shared_position[BLOCK_SIZE];
            float4 temp_position = p[tile*blockDim.x + threadIdx.x];
            shared_position[threadIdx.x] = make_float3(temp_position.x, temp_position.y, temp_position.z);
            __syncthreads();

            for (int j =0; j < BLOCK_SIZE; j++){
                float dx = shared_position[j].x - p[i].x;
                float dy = shared_position[j].y - p[i].y;
                float dz = shared_position[j].z - p[i].z;
                float distSqr = dx*dx + dy*dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist *invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
            __syncthreads();
        }
        v[i].x += dt * Fx;
        v[i].y += dt * Fy;
        v[i].z += dt * Fz;
    }        
}

int main(){
    int nBodies = 3e4;
    const float dt = 0.01f;
    const int nIters = 100;

    int size = 2 * nBodies * sizeof(float4);
    float *buf = (float*)malloc(size);
    NBodySystem p = { (float4*)buf, ((float4*)buf)+nBodies };

    //generateRandomizeBodies(buf, 8*nBodies);

    float *d_buf;
    cudaMalloc(&d_buf, size);
    NBodySystem d_p = { (float4*)buf, ((float4*)buf) + nBodies };

    int nBlocks = (nBodies + BLOCK_SIZE -1)/BLOCK_SIZE;
    for (int iter = 1; iter <= nIters; iter++){
        cudaMemcpy(d_buf, buf, size, cudaMemcpyDeviceToHost);
        calculateBodyForce<<< nBlocks, BLOCK_SIZE >>>(d_p.pos, d_p.vel, dt, nBodies);
        cudaMemcpy(buf, d_buf, size, cudaMemcpyDeviceToHost);

        for (int i =0 ; i< nBodies; i++){
            p.pos[i].x += p.vel[i].x * dt;
            p.pos[i].y += p.vel[i].y * dt;
            p.pos[i].z += p.vel[i].z * dt;
        }
        printf("Iteration %d \n", iter);
    }

    free(buf);
    cudaFree(d_buf);

    return 0;
}