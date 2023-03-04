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


int main(){
   
    return 0;
}