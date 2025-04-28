#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

// 生成测试用例
void m_reset(float** m, int n) {
    for (int i = 0; i < n; i++) {
        memset(m[i], 0, n * sizeof(float));
        m[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            m[i][j] = (float)rand() / RAND_MAX * 100.0f;
        }
    }

    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] += m[k][j];
            }
        }
    }
}

// 串行版本高斯消元
void gaussian_elimination_serial(float** A, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[k][k];
        int j;

        // Normalization
        for (j = k + 1; j < n; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;

        // Elimination
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            for (j = k + 1; j < n; j++) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0f;
        }
    }
}

// Neon向量化版本高斯消元
void gaussian_elimination_neon(float** A, int n) {
    for (int k = 0; k < n; k++) {
        float pivot = A[k][k];
        int j;

        // Normalization with Neon
        float32x4_t vt = vdupq_n_f32(pivot);
        for (j = k + 1; j + 4 <= n; j += 4) {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < n; j++) {
            A[k][j] /= pivot;
        }
        A[k][k] = 1.0f;

        // Elimination with Neon
        for (int i = k + 1; i < n; i++) {
            float factor = A[i][k];
            float32x4_t vaik = vdupq_n_f32(factor);

            for (j = k + 1; j + 4 <= n; j += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < n; j++) {
                A[i][j] -= A[k][j] * factor;
            }
            A[i][k] = 0.0f;
        }
    }
}

// 验证结果一致性
int verify(float** A, float** B, int n) {
    float epsilon = 1e-6;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A[i][j] - B[i][j]) > epsilon) {
                printf("Verification failed at (%d, %d): %f vs %f\n",
                    i, j, A[i][j], B[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// 计时函数
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
    const int N = 1024;
    const int trials = 5;

    // 分配对齐的内存
    float** A_serial = (float**)malloc(N * sizeof(float*));
    float** A_neon = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        posix_memalign((void**)&A_serial[i], 16, N * sizeof(float));
        posix_memalign((void**)&A_neon[i], 16, N * sizeof(float));
    }

    // 生成测试数据
    srand(time(NULL));
    m_reset(A_serial, N);
    for (int i = 0; i < N; i++) {
        memcpy(A_neon[i], A_serial[i], N * sizeof(float));
    }

    // 预热缓存
    gaussian_elimination_serial(A_serial, N);
    gaussian_elimination_neon(A_neon, N);

    // 正式测试
    double total_serial = 0, total_neon = 0;
    for (int t = 0; t < trials; t++) {
        // 重置测试数据
        m_reset(A_serial, N);
        for (int i = 0; i < N; i++) {
            memcpy(A_neon[i], A_serial[i], N * sizeof(float));
        }

        // 串行版本
        double start = get_time();
        gaussian_elimination_serial(A_serial, N);
        total_serial += get_time() - start;

        // Neon版本
        start = get_time();
        gaussian_elimination_neon(A_neon, N);
        total_neon += get_time() - start;

        // 验证结果
        if (!verify(A_serial, A_neon, N)) {
            printf("Result mismatch!\n");
            break;
        }
    }

    // 输出结果
    printf("Average Serial Time: %.6f s\n", total_serial / trials);
    printf("Average Neon Time:   %.6f s\n", total_neon / trials);
    printf("Speedup: %.6fX\n", total_serial / total_neon);

    // 释放内存
    for (int i = 0; i < N; i++) {
        free(A_serial[i]);
        free(A_neon[i]);
    }
    free(A_serial);
    free(A_neon);

    return 0;
}