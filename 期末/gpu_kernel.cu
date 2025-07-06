#include "gpu_kernel.h"
#define BLOCK_SIZE 256
#define GRID_SIZE(dim) ((dim + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void gpu_row_xor_kernel(uint64_t* matrix, uint64_t* eliminators, 
                                  int* row_flags, int total_rows, int col_blocks) {
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row_idx < total_rows && col_idx < col_blocks) {
        int global_idx = row_idx * col_blocks + col_idx;
        if (row_flags[row_idx]) {
            matrix[global_idx] ^= eliminators[col_idx];
        }
    }
}

void launch_gpu_pipeline(uint64_t* h_matrix, uint64_t* h_eliminators, 
                         int* h_row_flags, int total_rows, int col_blocks) {
    // 创建多流
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 分块处理
    const int chunk_size = total_rows / 10;
    for (int chunk = 0; chunk < 10; chunk++) {
        int offset = chunk * chunk_size;
        int cur_rows = min(chunk_size, total_rows - offset);
        
        // 异步传输
        cudaMemcpyAsync(..., cudaMemcpyHostToDevice, streams[chunk % 4]);
        
        // 内核配置
        dim3 blockDim(16, 16);
        dim3 gridDim(GRID_SIZE(col_blocks), GRID_SIZE(cur_rows));
        gpu_row_xor_kernel<<<gridDim, blockDim, 0, streams[chunk % 4]>>>(...);
        
        // 异步取回
        cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, streams[chunk % 4]);
    }
}