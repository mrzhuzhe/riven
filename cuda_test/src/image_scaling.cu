#include <stdio.h>
#include <srcImagePgmPpmPackage.h>

__global__ void createResizedImage(unsigned char *imageScaledData,
int scaled_width,
float scale_factor,
cudaTextureObject_t texObj
){
    const unsigned int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tidY = blockIdx.x * blockDim.x + threadIdx.y;
    const unsigned index = tidY * scaled_width + tidX;
    imageScaledData[index] = tex2D<unsigned char>(texObj, (float)(tidX*scale_factor),(float)(tidY*scale_factor));
}

int main(){
    int height=0, width=0, scaled_height=0, scaled_width=0;

    float scaling_ratio=0.5;
    unsigned char* data;
    unsigned char* scaled_data, *d_scaled_data;

    char inputStr[1024] = {"./src/aerosmith-double.pgm"};
    char outputStr[1024] = {"./build/aerosmith-double-scaled.pgm"};

    cudaError_t returnValue;

    cudaArray* cu_array;
    cudaChannelFormatKind kind = cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, kind);

    get_PgmPpmParams(inputStr, &height, &width);
    data = (unsigned char*)malloc(height*width*sizeof(unsigned char));
    printf("\n Reading image width height [%d][%d]", height, width);
    scr_read_pgm(inputStr, data, height, width);

    scaled_height = (int)(height*scaling_ratio);
    scaled_width = (int)(width*scaling_ratio);
    scaled_data = (unsigned char*)malloc(scaled_height*scaled_width*sizeof(unsigned char));
    printf("\n scaled image width height [%d][%d]", scaled_height, scaled_width);

    returnValue = cudaMallocArray(&cu_array, &channelDesc, width, height);
    //returnValue = (cudaError_t)(returnValue | cudaMemcpy(cu_array, data, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice));
   
    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    const size_t spitch = width * sizeof(unsigned char);
    // Copy data located at address h_data in host memory to device memory

    //  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    returnValue = (cudaError_t)(returnValue | cudaMemcpy2DToArray(cu_array, 0, 0, data, spitch, width * sizeof(unsigned char),
                        height, cudaMemcpyHostToDevice));

    if(returnValue != cudaSuccess)
		printf("\n Got error while running CUDA API Array Copy");
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_array;

    struct cudaTextureDesc texDesc;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    returnValue = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    if(returnValue != cudaSuccess)
      printf("\n Got error while running CUDA API Bind Texture");

    cudaMalloc(&d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char));

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(scaled_width/dimBlock.x, scaled_height/dimBlock.y, 1);
    printf("\n Launching grid with blocks [%d][%d]", dimGrid.x , dimGrid.y);

    createResizedImage<<<dimGrid, dimBlock>>>(d_scaled_data, scaled_width, 1/scaling_ratio, texObj);

    returnValue = (cudaError_t)(returnValue | cudaDeviceSynchronize());   
    returnValue = (cudaError_t)(returnValue | cudaMemcpy(scaled_data, d_scaled_data, scaled_height*scaled_width*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    if (returnValue != cudaSuccess)
      printf("\n Got error while running CUDA API kernel");

    cudaDestroyTextureObject(texObj);

    scr_write_pgm(outputStr, scaled_data, scaled_height, scaled_width, "###");

    if (data != NULL)
      free(data);
    if(cu_array != NULL)
      cudaFreeArray(cu_array);
    if(scaled_data != NULL)
      free(scaled_data);
    if(d_scaled_data!=NULL)
      cudaFree(d_scaled_data);

    return 0;
}