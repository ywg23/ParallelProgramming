#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>

#define N 512
#define NUM_THREADS 4

typedef float ele_t;

void init_matrix(ele_t mat[N][N]) {
    srand(time(0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = (i == j) ? 1.0f : (rand() % 10 + 1);
}

void lu_serial(ele_t mat[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < N; ++k) {
                mat[j][k] -= div * mat[i][k];
            }
        }
    }
}
void lu_omp_simple(ele_t mat[N][N]) {
    for (int i = 0; i < N; ++i) {
#pragma omp parallel for
        for (int j = i + 1; j < N; ++j) {
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < N; ++k) {
                mat[j][k] -= div * mat[i][k];
            }
        }
    }
}
void lu_omp_static(ele_t mat[N][N]) {
    int chunk = N / NUM_THREADS;
    for (int i = 0; i < N; ++i) {
#pragma omp parallel for schedule(static, chunk)
        for (int j = i + 1; j < N; ++j) {
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < N; ++k) {
                mat[j][k] -= div * mat[i][k];
            }
        }
    }
}
void lu_omp_dynamic_block(ele_t mat[N][N], int B) {
    int chunk = N / NUM_THREADS / B;
    if (chunk < 1) chunk = 1;
    for (int i = 0; i < N; ++i) {
#pragma omp parallel for schedule(dynamic, chunk)
        for (int j = i + 1; j < N; ++j) {
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < N; ++k) {
                mat[j][k] -= div * mat[i][k];
            }
        }
    }
}
void lu_omp_guided(ele_t mat[N][N]) {
    for (int i = 0; i < N; ++i) {
#pragma omp parallel for schedule(guided)
        for (int j = i + 1; j < N; ++j) {
            ele_t div = mat[j][i] / mat[i][i];
            for (int k = i; k < N; ++k) {
                mat[j][k] -= div * mat[i][k];
            }
        }
    }
}


int main() {
    omp_set_num_threads(NUM_THREADS);

    ele_t mat[N][N], mat_copy[N][N];
    double start, end;

    // 1. 串行
    init_matrix(mat);
    memcpy(mat_copy, mat, sizeof(mat));
    start = omp_get_wtime();
    lu_serial(mat_copy);
    end = omp_get_wtime();
    std::cout << "1. Serial time: " << end - start << " s\n";

    // 2. 简单并行
    memcpy(mat_copy, mat, sizeof(mat));
    start = omp_get_wtime();
    lu_omp_simple(mat_copy);
    end = omp_get_wtime();
    std::cout << "2. Simple parallel time: " << end - start << " s\n";

    // 3. 静态调度
    memcpy(mat_copy, mat, sizeof(mat));
    start = omp_get_wtime();
    lu_omp_static(mat_copy);
    end = omp_get_wtime();
    std::cout << "3. Static schedule time: " << end - start << " s\n";

    // 4. 动态调度，块数 B=2 例如
    memcpy(mat_copy, mat, sizeof(mat));
    start = omp_get_wtime();
    lu_omp_dynamic_block(mat_copy, 2);
    end = omp_get_wtime();
    std::cout << "4. Dynamic schedule (block=2) time: " << end - start << " s\n";

    // 5. 递减调度
    memcpy(mat_copy, mat, sizeof(mat));
    start = omp_get_wtime();
    lu_omp_guided(mat_copy);
    end = omp_get_wtime();
    std::cout << "5. Guided schedule time: " << end - start << " s\n";

    return 0;
}
