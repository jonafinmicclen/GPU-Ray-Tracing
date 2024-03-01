#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// CUDA kernel for vector addition
__global__ void addKernel(int* c, const int* a, const int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// Host function to launch the CUDA kernel
cudaError_t addWithCuda(int* c, const int* a, const int* b, int size) {
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (a, b, and c)
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_a);
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        return cudaStatus;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return cudaStatus;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return cudaStatus;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        return cudaStatus;
    }

    // Free GPU memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return cudaStatus;
}