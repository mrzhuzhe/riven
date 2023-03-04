#include "stb_image.h"
#include "stb_image_write.h"
#include "allocator.h"
#include "TikTok.h"
#include <cuda_runtime.h>

template <class A>
std::tuple<int, int, int> read_image(A &a, const char *path){
    int nx = 0, ny = 0, comp = 0;
    unsigned char *p = stbi_load(path, &nx, &ny, &comp, 0);
    if (!p){
        perror(path);
        exit(-1);
    }
    a.resize(nx*ny*comp);
    for (int c = 0; c < comp; c++){
        for (int y = 0; y < ny; y++){
            for (int x = 0; x < nx; x++){
                a[c*nx*ny+y*nx+x] = (1.f/255.f) * p[(y*nx + x)*comp+c];
            }
        }
    }
    stbi_image_free(p);
    return {nx, ny, comp};
}

template <class A>
void write_image(A const &a, int nx, int ny, int comp, const char *path){
    auto p = (unsigned char *)malloc(nx*ny*comp);
    for (int c =0; c < comp; c++){
        for (int y =0; y <ny; y++){
            for (int x=0; x< nx; x++){
                p[(y*nx+x)*comp + c] = std::max(0.f, std::min(255.f, a[c*nx*ny + y*nx +x]*255.f));
            }
        }
    }
    int ret = 0;
    auto pt = strrchr(path, '.');
    if (pt && !strcmp(pt, ".png")){
        ret = stbi_write_png(path, nx, ny, comp, p, 0);
    } else if (pt && !strcmp(pt, ".jpg")){
        ret = stbi_write_jpg(path, nx, ny, comp, p, 0);
    } else {
        ret = stbi_write_bmp(path, nx, ny, comp, p);
    }
    free(p);
    if (!ret){
        perror(path);
        exit(-1);
    }
}

template <int nblur, int blockSize>
__global__ void parallel_xblur(float *out, float const *in, int nx, int ny){
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x>=nx || y>=ny) return;
    float sum = 0;
    for (int i =0; i<nblur; i++){
        sum += in[y *nx+ std::min(x+i, nx-1)];
    } 
    out[y*nx+x] = sum / nblur;
}

template <int blockSize>
__global__ void parallel_jacobi(float *out, float const *in, int nx, int ny){
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;
    if (x>=nx || y>=ny) return;
    float res = (
        in[std::min(y+1, ny-1)*nx + x]
        +in[std::max(y-1, 0)*nx +x]
        +in[y*nx+std::min(x+1, nx-1)]
        +in[y*nx+std::max(x-1, 0)]
    ) / 4;
    out[y*nx+x] = res;
}

template <int iters, int blockSize>
__global__ void parallel_jacobi_kernel(float *out, float const *in, int nx, int ny){
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    constexpr int chunkSize = blockSize - iters * 2;
    int globalX = blockX * chunkSize - iters + threadX;
    int globalY = blockY * chunkSize - iters + threadX;

    __shared__ float mem[2][blockSize+2][blockSize+2];
    int clampedX = std::min(std::max(globalX, 0), nx-1);
    int clampedY = std::min(std::max(globalY, 0), ny-1);

    mem[0][1+threadY][1+threadX] = in[nx*clampedY + clampedX];

    if (threadY == 0){
        int clampedYn = std::min(std::max(blockY * chunkSize - iters - 1, 0), ny - 1);
        mem[0][0][1+threadX] = in[nx*clampedYn + clampedX];
        int clampedYp = std::min(std::max(blockY * chunkSize - iters + blockSize, 0), ny-1);
        mem[0][1+blockSize][1+threadX] = in[nx*clampedYp + clampedX];
    }

    if (threadX == 0){
        int clampedXn = std::min(std::max(blockX * chunkSize - iters - 1, 0), nx - 1);
        mem[0][0][1+threadY] = in[nx*clampedY + clampedXn];
        int clampedXp = std::min(std::max(blockX * chunkSize - iters + blockSize, 0), nx-1);
        mem[0][1+threadY][1+blockSize]= in[nx*clampedY + clampedXp];
    }

    __syncthreads();

    for (int stage = 0; stage < iters; stage += 2){
#pragma unroll
        for (int phase = 0; phase < 2; phase ++){
            mem[1-phase][1+threadY][1+threadX] = 
                ( mem[phase][1 + threadY + 1][1+ threadX]
                + mem[phase][1 + threadY - 1][1+ threadX]
                + mem[phase][1 + threadY][1 + threadX + 1]
                + mem[phase][1 + threadY][1 + threadX - 1]
                ) /4;
            __syncthreads();
        }
    }

    if (threadX >= iters && threadX < blockSize - iters){
        if (threadY >= iters && threadY < blockSize - iters){
            if (globalX < nx && globalY < ny){
                out[globalY*nx + globalX] = mem[0][1+threadY][1+threadX];
            }
        }
    }



}

template <int iters, int blockSize>
void parallel_jacobi2(float *out, float const *in, int nx, int ny){
    constexpr int chunkSize = blockSize - iters * 2;
    static_assert(chunkSize > 0 && iters % 2 == 0);
    parallel_jacobi_kernel<iters, blockSize>
    <<<dim3((nx+chunkSize-1)/ chunkSize, (ny+chunkSize-1)/chunkSize, 1), dim3(blockSize, blockSize, 1)>>>
    (out, in, nx, ny);
}


int main(){
    std::vector<float, CudaAllocator<float>> in;
    std::vector<float, CudaAllocator<float>> out;

    auto [nx, ny, _] = read_image(in, "./outputs/original.jpg");
    out.resize(in.size());

    Tik();
    /*
    parallel_xblur<32, 32><<<dim3((nx+31)/32, (ny+31)/32, 1), dim3(32, 32, 1)>>>
    (out.data(), in.data(), nx, ny);
    */
    /*
    for (int step = 0; step < 1024; step++){
        parallel_jacobi<32><<<dim3((nx+31)/32, (ny+31)/32, 1), dim3(32, 32, 1)>>>
            (out.data(), in.data(), nx, ny);
        std::swap(out, in);
    }
    */
    constexpr int iters = 4;
    for (int step = 0; step < 1024; step += iters){
        parallel_jacobi2<iters, 32>
            (out.data(), in.data(), nx, ny);
        std::swap(out, in);
    }


    cudaDeviceSynchronize();
    Tok("blur");
    write_image(out, nx, ny, 1, "./outputs/out.png");
    //system("display ./outputs/out.png &");

    return 0;
}