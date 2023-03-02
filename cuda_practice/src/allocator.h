template <class T>
struct CudaAllocator {
    using value_type = T;

    T *allocate(size_t size) {
        T *ptr = nullptr;
        cudaMallocManaged(&ptr, size * sizeof(T));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0) {
        cudaFree(ptr);
    }

    template <class ...Args>
    void construct(T *p, Args &&...args){
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>))::new((void *)p) T(std::forward<Args>(args)...);        
    }
};