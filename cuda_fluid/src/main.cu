#include <cstdio>
#include <cuda_runtime.h>

struct DisableCopy {
    DisableCopy() = default;
    DisableCopy(DisableCopy const &) = delete;
    DisableCopy &operator=(DisableCopy const &) = delete;
}

template <class T>
struct  CudaArray : DisableCopy
{
    cudaArray *m_cuArray{};
    uint3 m_dim{};
    explicit CudaArray(uint3 const &_dim): m_dim(_dim){
        cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        cudaMalloc3DArray(&m_cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore);
    }

    void copyIn(T const *_data){
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcPtr = make_cudaPitchedPtr((void *)_data, m_dim.x* sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.dstArray = m_cuArray;
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3Dparams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copy3DParams);
    }

    void copyOut(T *_data){
        cudaMemcpy3DParms copy3DParams{};
        copy3DParams.srcArray = m_cuArray;
        copy3DParams.dstPtr = make_cudaPitchedPtr((void *)_data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copy3DParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copy3Dparams.kind = cudaMemcpyDeviceToHost;
        cudaMemcpy3D(&copy3DParams);
    }

    cudaArray *getArray() const {
        return m_cuArray;
    }

    ~CudaArray(){
        cudaFreeArray(m_cuArray);
    }

};

template <class T>
struct CudaSurfaceAccessor {
    cudaSurfaceObject_t m_cuSuf;
    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T read(int x, int y, int z) const {
        return surf3Dread<T>(m_cuSuf, x*sizeof(T), y, z, mode);
    }
    template <cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T write(int x, int y, int z) const {
        return surf3Dwrite<T>(val, m_cuSuf, x*sizeof(T), y, z, mode);
    }
};

template <class T>
struct  CudaSurface : CudaArray<T>
{
    cudaSurfaceObject_t m_cuSur{};
    explicit CudaSurface(uint3 const &_dim): CudaArray<T>(_dim){
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaArray<T>::getArray();
        cudaCreateSurfaceObject(&m_cuSuf, &resDesc);
    }  

    cudaSurfaceObject_t getSurface() const {
        return m_cuSuf;
    }

    CudaSurfaceAccessor<T> accessSurface() const {
        return {m_cuSuf}
    }
    ~CudaSurface(){
        cudaDEstroySurfaceObject(m_cuSuf);
    }

};


template <class T>
struct CudaTextureAccessor {
    cudaTextureObject_t m_cuTex;
    __device__ __forceinline__ T sample(float x, float y, foat z) const {
        return tex3D<T>(m_cuTex, x, y, z);
    }
};

template <class T>
struct  CudaTexture : CudaSurface<T>{
    struct  Parameters {
        cudaTextureAddressMode addressMode{cudaAddressModeClamp};
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};
        cudaTextureReadMode readMode{cudaReadModeElementType};
        bool normalizedCoords{false};
    };

    cudaTextureObject_t m_cuTex{};

    explicit CudaTexture(uint3 const &_dim, Parameters const &_args = {}): CudaSurface<T>(_dim){
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaSurface<T>::getArray();

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = _args.addressMode;
        texDesc.addressMode[1] = _args.addressMode;
        texDesc.addressMode[2] = _args.addressMode;
        texDesc.filterMode = _args.filterMode;
        texDesc.readMode = _args.readMode;
        texDesc.normalizedCoords = _args.normalizedCoords;

        cudaCreateTextureObject(&m_cuTex, &resDesc, &texDesc, NULL);

    }

    cudaTextureObject_t getTexture() const {
        return m_cuTex;
    }

    CudaTextureAccessor<T> accessTexture() const {
        return {m_cuTex};
    }

    ~CudaTexture(){
        cudaDestroyTextureObject(m_cuTex);
    }


};

struct SmokeSim : DisableCopy
{
    unsinged int n;
    std::unique_ptr<CudaSurface<float4>> loc;
    std::unique_ptr<CudaTexture<float4>> vel;
    std::unique_ptr<CudaTexture<float4>> velNext;
    std::unique_ptr<CudaTexture<float4>> clr;
    std::uniqur_ptr<CudaTexture<float4>> clrNext;

    std::unique_ptr<CudaSurface<float>> div;
    std::unique_ptr<CudaSurface<float>> pre;
    std::vector<std::unique_ptr<CudaSurface<float>>> res;
    std::vector<std::unique_ptr<CudaSurface<float>>> res2;
    std::vector<std::unique_ptr<CudaSurface<float>>> err2;
    std:vector<unsigned int> sizes;

    explicit SmokeSim(unsigned int _n, unsigned int _n0 = 16)
    :n(_n), loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n}))
    , vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , velNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , clr(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , clrNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
    , div(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    , pre(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
    {
        unsigned int tn;
        for (tn = n; tn >= _n0; tn /=2){
            res.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
            res2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            err2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            sizes.push_back(tn);
        }
    }
};

__global__ void advect_kernel(CudaTextureAccessor<float4> texVel, CudaSurfaceAccessor<float4> sufloc, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;

    auto sample = [] (CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
        float4 vel = tex.sample(loc.x, loc.y, loc.z);
        return make_float3(vel.x, vel.y, vel.z);
    };

    float loc = make_float3(x+0.5f, y + 0.5f, z+0.5f);
    float3 vel1 = sample(texVel, loc);
    float3 vel2 = sample(texVel, loc-0.5f*vel1);
    float3 vel3 = sample(texVel, loc-0.5f*vel2);
    loc -= (2.f/9.f) * vel1 + (1.f/3.f)*vel2 + (4.f/9.f)*vel3;
    surLoc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);

}

__global__ void resample_kernel(CudaTextureAccessor<float4> sufLoc, CudaSurfaceAccessor<float4> texClr, CudaSurfaceAccessor<float4> sufClrNext, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;

    float4 loc = sufLoc.read(x, y, z);
    float4 clr = texClr.sample(loc.x, loc.y, loc.z);
    sufClrNext.write(clr, x, y, z);
}

void advection() {
    advect_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(vel->accessTexture(), loc->accessSurface(), n);
    resample_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(loc->accessSurface(), clr->accessTexture(), clrNext->accessSurface(), n);
    resample_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(loc->accessSurface(), vel->accessTexture(), velNext->accessSurface(), n);

    std::swap(vel, velNext);
    std::swap(clr, clrNext);
}

void step(int times = 16){
    for (int step = 0; step < times, step++){
        projection();
        advection();
    }
}

__global__ void divergence_kernel(CudaSurfaceAccessor<float4> sufVel, CudaSurfaceAccessor<float> surDiv, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;

    float vxp = sufVel.read<cudaBoundaryModeZero>(x+1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeZero>(x, y+1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeZero>(x, y, z+1).z;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x-1, y, z).x;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x, y-1, z).y;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x, y, z-1).z;
    float div = (vxp - vxn + vyp - vyn + vzp - vzn) * 0.5f;
    sufDiv.write(div, x, y, z);
}

__global__ void jacobi_kernel(CudaSurfaceAccessor<float> sufDiv, CudaSurfaceAccessor<float> surPre, CudaSurfaceAccessor<float> surPreNext, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;

    float pxp = sufVel.read<cudaBoundaryModeClamp>(x+1, y, z);
    float pxn = sufVel.read<cudaBoundaryModeClamp>(x-1, y, z);
    float pyp = sufVel.read<cudaBoundaryModeClamp>(x, y+1, z);
    float pyn = sufVel.read<cudaBoundaryModeClamp>(x, y-1, z);
    float pzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z+1);
    float pzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z-1);
    float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
    sufPreNext.write(preNext, x, y, z);
}


__global__ void subgradient_kernel(CudaSurfaceAccessor<float> sufPre, CudaSurfaceAccessor<float> sufVel, CudaSurfaceAccessor<float> sufBound, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;
    if (sufBound.read(x, y, z) < 0) return;

    float vxp = sufVel.read<cudaBoundaryModeZero>(x+1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeZero>(x, y+1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeZero>(x, y, z+1).z;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x-1, y, z).x;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x, y-1, z).y;
    float vxn = sufVel.read<cudaBoundaryModeZero>(x, y, z-1).z;
    float vel = sufVel.read(x, y,z);
    vel.x -= (pxp - pxn) * 0.5f;
    vel.y -= (pyp - pyn) * 0.5f;
    vel.z -= (pzp - pzn) * 0.5f;
    sufVel.write(vel, x, y, z);
}

void projection (int times = 400){
    divergence_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(vel->accessSurface(), div->accessSurface(), n);

    for (int step = 0; step < times, step++){
        jacobi_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(div->accessSurface(), pre->accessSurface(), preNext->accessSurface(), n);
        std::swap(pre, preNext);
    }

    subgradient_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(pre->accessSurface(), vel->accessSurface(), n);
}

float calc_loss(){
    divergence_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(vel->accessSurface(), div->accessSurface(), n);
    float *sum;
    cudaMalloc(&sum, sizeof(float));
    sumloss_kernel<<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(div->accessSurface(), sum, n);
    float *sum;
    float cpu;
    cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sum);
    return cpu;
}

__global__ void sumloss_kernel(CudaSurfaceAccessor<float> sufDiv, float *sum, unsigned int n){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >=n || z>=n ) return;

    float div = sufDiv.read(x, y, z);
    atomicAdd(sum, div*div);
}

void smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times = 4){
    usigned int tn = sizes[lev];
    for (int step = 0; step < times; step++){
        rgbs_kernel<0><<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
        rgbs_kernel<1><<<dim3((n+7)/8, (n+7)/8, (n+7)/8), dim3(8, 8, 8)>>>(v->accessSurface(), f->accessSurface(), tn);
    }
}

void vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f){
    if (lev >= sizes.size()){
        unsigned int tn = sizes.back() / 2;
        smooth(v, f, lev);
        return;
    }
    auto *r = res[lev].get();
    auto *r2 = res2[lev].get();
    auto *e2 = err2[lev].get();
    unsigned int tn = sizes[lev];
    smooth(v, f, lev);
    residual_kernel<<<dim3((tn+7)/8, (tn+7)/8, (tn+7)/8), dim3(8, 8, 8)>>>(r->accessSurface(), v->accessSurface(), f->accessSurface(), tn);
    restrict_kernel<<<dim3((tn/2+7)/8, (tn/2+7)/8, (tn/2+7)/8), dim3(8, 8, 8)>>>(r2->accessSurface(), r->accessSurface(), tn/2);
    fillzero_kernel<<<dim3((tn/2+7)/8, (tn/2+7)/8, (tn/2+7)/8), dim3(8, 8, 8)>>>(e2->accessSurface(), tn/2);    
    vcycle(lev+1, e2, r2);
    prolongate_kernel<<<dim3((tn/2+7)/8, (tn/2+7)/8, (tn/2+7)/8), dim3(8, 8, 8)>>>(v->accessSurface(), e2->accessSurface(), tn/2);    
    smooth(v, f, lev);
}

int main(){
   
    return 0;
}