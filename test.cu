#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cudnn.h>
#include <cuda_fp16.h>

using namespace std::chrono;

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


#define checkCUDNN(expression)                             \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

void print_array(float *array, int size, const char *name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void init_array(float *array, int size, float val) {
  for (int i = 0; i < size; i++) {
    array[i] = val;
  }
}


void test(){
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStream_t stream_1;
    cudaStreamCreate(&stream_1);

    CUmemoryPool pool_;
    cuDeviceGetDefaultMemPool(&pool_, 0);
    uint64_t threshold = UINT64_MAX;
    cuMemPoolSetAttribute(pool_, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);

    int _n = 128, _c= 2048, _h = 7, _w = 7;

    int x_size = _n * _c * _h * _w;
    int x_size_bytes = x_size * sizeof(float);
    int iter = 8;
    int mean_size = _c;
    int mean_size_bytes = mean_size * sizeof(float);
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool graphCreated=false;
    float* h_x = (float*)malloc(x_size_bytes);
    float* h_y = (float*)malloc(x_size_bytes);
    init_array(h_x, x_size, 2.5);
    init_array(h_y, x_size, 0.0);
    float *x, *y;
    checkCUDA(cudaMalloc(&x, x_size_bytes));
    checkCUDA(cudaMalloc(&y, x_size_bytes));
    cudaMemcpy(x, reinterpret_cast<const float *>(h_x), x_size_bytes, cudaMemcpyHostToDevice);
    float *scale, *offset;
    float *saved_mean, *saved_inv_var;
    float* h_scale = (float*)malloc(mean_size_bytes);
    float* h_offset = (float*)malloc(mean_size_bytes);
    init_array(h_scale, mean_size, 1.5);
    init_array(h_offset, mean_size, 2.0);
    checkCUDA(cudaMallocManaged(&scale, mean_size_bytes));
    checkCUDA(cudaMallocManaged(&offset, mean_size_bytes));
    cudaMemcpy(scale, reinterpret_cast<const float *>(h_scale), mean_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(offset, reinterpret_cast<const float *>(h_offset), mean_size_bytes, cudaMemcpyHostToDevice);
    checkCUDA(cudaMallocManaged(&scale, mean_size_bytes));
    checkCUDA(cudaMallocManaged(&offset, mean_size_bytes));
    checkCUDA(cudaMallocManaged(&saved_mean, mean_size_bytes));
    checkCUDA(cudaMallocManaged(&saved_inv_var, mean_size_bytes));
    float *a_x, *a_y, *a_scale, *a_offset, *a_saved_mean, *a_saved_inv_var;
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    for (int i =0; i < iter; i++){
        if (!graphCreated){
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_x), x_size_bytes, pool_, stream);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_y), x_size_bytes, pool_, stream);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_scale), mean_size_bytes, pool_, stream);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_offset), mean_size_bytes, pool_, stream);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_saved_mean), mean_size_bytes, pool_, stream);
          cuMemAllocFromPoolAsync(reinterpret_cast<CUdeviceptr*>(&a_saved_inv_var), mean_size_bytes, pool_, stream);
          cudaMemcpyAsync(a_x, x, x_size_bytes, cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(a_y, y, x_size_bytes, cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(a_scale, scale, mean_size_bytes, cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(a_offset, offset, mean_size_bytes, cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(a_saved_mean, saved_mean, mean_size_bytes, cudaMemcpyDeviceToDevice, stream);
          cudaMemcpyAsync(a_saved_inv_var, saved_inv_var, mean_size_bytes, cudaMemcpyDeviceToDevice, stream);
          auto mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
          float one = 1.0;
          float zero = 0.0;
          //int N = 128, C = 2048, H = 7, W = 7;
          cudnnTensorDescriptor_t x_descriptor;
          checkCUDNN(cudnnCreateTensorDescriptor(&x_descriptor));
          checkCUDNN(cudnnSetTensor4dDescriptor(x_descriptor,
                                                /*format=*/CUDNN_TENSOR_NHWC,
                                                /*dataType=*/CUDNN_DATA_FLOAT,
                                                /*batch_size=*/128,
                                                /*channels=*/2048,
                                                /*image_height=*/7,
                                                /*image_width=*/7));
          cudnnTensorDescriptor_t mean_descriptor;
          checkCUDNN(cudnnCreateTensorDescriptor(&mean_descriptor));
          checkCUDNN(cudnnSetTensor4dDescriptor(mean_descriptor,
                                                /*format=*/CUDNN_TENSOR_NHWC,
                                                /*dataType=*/CUDNN_DATA_FLOAT,
                                                /*batch_size=*/1,
                                                /*channels=*/2048,
                                                /*image_height=*/1,
                                                /*image_width=*/1));
            checkCUDNN(cudnnBatchNormalizationForwardInference(
                  /*handle=*/cudnn,
                  /*mode=*/mode,
                  /*alphaDataDiff=*/&one,
                  /*betaDataDiff=*/&zero,
                  /*xDesc=*/x_descriptor,
                  a_x,
                  /*xDesc=*/x_descriptor,
                  a_y,
                  /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
                  /*bnScale=*/a_scale,
                  /*bnBias=*/a_offset,
                  /*resultSaveMean=*/a_saved_mean,
                  /*resultSaveInvVariance=*/a_saved_inv_var,
                  /*epsilon=*/0.001)
             )

            cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_x), stream);
            cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_scale), stream);
            cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_offset), stream);
            cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_saved_mean), stream);
            cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_saved_inv_var), stream);
            checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
            checkCudaErrors(cudaGraphUpload(instance, stream));
            graphCreated = true;
        }
        checkCudaErrors(cudaGraphLaunch(instance, stream));
        checkCUDA(cudaDeviceSynchronize());
        float* out = (float*)malloc(x_size_bytes);
        cudaMemcpy(out, reinterpret_cast<const float *>(a_y), x_size_bytes, cudaMemcpyDeviceToHost);
        cuMemFreeAsync(reinterpret_cast<const CUdeviceptr&>(a_y), stream);
        print_array(out, x_size, "dx NCHW format: ");
    }

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    checkCUDA(cudaFree(x));
    checkCUDA(cudaFree(y));
    checkCUDA(cudaFree(scale));
    checkCUDA(cudaFree(offset));
    checkCUDA(cudaFree(saved_mean));
    checkCUDA(cudaFree(saved_inv_var));
    free(h_x);
    free(h_y);
    free(h_scale);
    free(h_offset);
}

int main() {
    test();
    return 0;
}
