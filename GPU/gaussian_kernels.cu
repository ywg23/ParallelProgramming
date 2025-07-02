#include "utils.h"

// 基础版除法核函数
__global__ void division_kernel_basic(float* data, int k, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float pivot = data[k * N + k];
        data[k * N + tid] /= pivot;
    }
}

// 基础版消去核函数
__global__ void eliminate_kernel_basic(float* data, int k, int N) {
    int tid = threadIdx.x;
    int row = k + 1 + blockIdx.x;

    while (row < N) {
        float factor = data[row * N + k];
        for (int col = k + 1 + tid; col < N; col += blockDim.x) {
            data[row * N + col] -= factor * data[k * N + col];
        }
        __syncthreads();

        // 主元下方置零（由线程0完成）
        if (tid == 0) {
            data[row * N + k] = 0.0f;
        }
        __syncthreads();

        row += gridDim.x;
    }
}

// 优化版除法核函数（使用向量化）
__global__ void division_kernel_opt(float* data, int k, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float pivot = data[k * N + k];

    // 使用网格跨步处理所有列
    for (int col = tid; col < N; col += stride) {
        data[k * N + col] /= pivot;
    }
}

// 优化版消去核函数（使用共享内存和向量化）
__global__ void eliminate_kernel_opt(float* data, int k, int N) {
    extern __shared__ float shared_row[]; // 动态共享内存，用于存储主元行
    int tid = threadIdx.x;
    int row = k + 1 + blockIdx.x;

    // 协作加载主元行（从第k列开始）
    for (int i = tid; i < N - k; i += blockDim.x) {
        shared_row[i] = data[k * N + k + i];
    }
    __syncthreads();

    // 网格跨步处理多行
    while (row < N) {
        float factor = data[row * N + k];

        // 向量化处理列（每次处理4个元素）
        int col = k + 1 + tid * 4;
        while (col < N - 3) {
            float4 val = *reinterpret_cast<float4*>(&data[row * N + col]);
            float4 pivot_val = *reinterpret_cast<float4*>(&shared_row[col - k]);
            
            val.x -= factor * pivot_val.x;
            val.y -= factor * pivot_val.y;
            val.z -= factor * pivot_val.z;
            val.w -= factor * pivot_val.w;
            
            *reinterpret_cast<float4*>(&data[row * N + col]) = val;
            col += blockDim.x * 4;
        }

        // 处理剩余元素（无法向量化的部分）
        for (int c = col; c < N; c += blockDim.x) {
            data[row * N + c] -= factor * shared_row[c - k];
        }

        __syncthreads();

        // 主元下方置零
        if (tid == 0) {
            data[row * N + k] = 0.0f;
        }
        __syncthreads();

        row += gridDim.x;
    }
}