#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// 检查CUDA运行时错误的宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) - %s\n", \
                __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 生成一个随机矩阵（对角占优，避免主元为0）
float* generate_matrix(int N) {
    float* matrix = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (float)(rand() % 100) / 100.0f; // [0,1)
        }
        // 使对角元素占优
        matrix[i * N + i] += 100.0f;
    }
    return matrix;
}

// 复制矩阵
float* copy_matrix(const float* src, int N) {
    float* dst = (float*)malloc(N * N * sizeof(float));
    memcpy(dst, src, N * N * sizeof(float));
    return dst;
}

// 验证结果（比较CPU和GPU的结果）
void verify_results(const float* cpu, const float* gpu, int N, const char* label) {
    float max_error = 0.0f;
    float max_relative_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float diff = fabs(cpu[i * N + j] - gpu[i * N + j]);
            if (diff > 1e-5) {
                error_count++;
            }
            if (diff > max_error) max_error = diff;

            // 计算相对误差（避免除以0）
            if (fabs(cpu[i * N + j]) > 1e-5) {
#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

                // 检查CUDA运行时错误的宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) - %s\n", \
                __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// 生成一个随机矩阵（对角占优，避免主元为0）
                float* generate_matrix(int N) {
                    float* matrix = (float*)malloc(N * N * sizeof(float));
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            matrix[i * N + j] = (float)(rand() % 100) / 100.0f; // [0,1)
                        }
                        // 使对角元素占优
                        matrix[i * N + i] += 100.0f;
                    }
                    return matrix;
                }

                // 复制矩阵
                float* copy_matrix(const float* src, int N) {
                    float* dst = (float*)malloc(N * N * sizeof(float));
                    memcpy(dst, src, N * N * sizeof(float));
                    return dst;
                }

                // 验证结果（比较CPU和GPU的结果）
                void verify_results(const float* cpu, const float* gpu, int N, const char* label) {
                    float max_error = 0.0f;
                    float max_relative_error = 0.0f;
                    int error_count = 0;

                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            float diff = fabs(cpu[i * N + j] - gpu[i * N + j]);
                            if (diff > 1e-5) {
                                error_count++;
                            }
                            if (diff > max_error) max_error = diff;

                            // 计算相对误差（避免除以0）
                            if (fabs(cpu[i * N + j]) > 1e-5) {
                                float relative_error = diff / fabs(cpu[i * N + j]);
                                if (relative_error > max_relative_error) max_relative_error = relative_error;
                            }
                        }
                    }

                    printf("%s验证结果: 最大绝对误差=%.2e, 最大相对误差=%.2e, 错误元素数=%d/%d (%.2f%%)\n",
                        label, max_error, max_relative_error, error_count, N * N,
                        100.0f * error_count / (N * N));
                }

                // 打印矩阵（用于调试）
                void print_matrix(const float* matrix, int N, const char* name) {
                    printf("%s矩阵:\n", name);
                    for (int i = 0; i < N; i++) {
                        for (int j = 0; j < N; j++) {
                            printf("%8.4f ", matrix[i * N + j]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                }

                // 获取当前时间（毫秒）
                double get_time_ms() {
                    struct timeval tv;
                    gettimeofday(&tv, NULL);
                    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
                }

#endif // UTILS_H      float relative_error = diff / fabs(cpu[i * N + j]);
                if (relative_error > max_relative_error) max_relative_error = relative_error;
            }
        }
    }

    printf("%s验证结果: 最大绝对误差=%.2e, 最大相对误差=%.2e, 错误元素数=%d/%d (%.2f%%)\n",
        label, max_error, max_relative_error, error_count, N * N,
        100.0f * error_count / (N * N));
}

// 打印矩阵（用于调试）
void print_matrix(const float* matrix, int N, const char* name) {
    printf("%s矩阵:\n", name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.4f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// 获取当前时间（毫秒）
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#endif // UTILS_H