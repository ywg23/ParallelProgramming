#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define MAX_N 1024
#define NUM_THREADS 8

float A[MAX_N][MAX_N];
int n = 1024;  // 矩阵规模

void init_matrix() {
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            A[i][j] = (float)(rand() % 100 + 1);
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            float ratio = rand() % 5;
            for (int j = 0; j < n; ++j)
                A[i][j] += ratio * A[k][j];
        }
    }
}

int main() {
    init_matrix();
    struct timeval start, end;
    gettimeofday(&start, NULL);

    int i, j, k;
    float tmp;

    #pragma omp parallel private(i, j, k, tmp) num_threads(NUM_THREADS)
    for (k = 0; k < n; ++k) {
        #pragma omp single
        {
            tmp = A[k][k];
            for (j = k + 1; j < n; ++j)
                A[k][j] /= tmp;
            A[k][k] = 1.0;
        }

        #pragma omp for schedule(static)
        for (i = k + 1; i < n; ++i) {
            tmp = A[i][k];
            for (j = k + 1; j < n; ++j)
                A[i][j] -= tmp * A[k][j];
            A[i][k] = 0.0;
        }
    }

    gettimeofday(&end, NULL);
    double time = 1000.0 * (end.tv_sec - start.tv_sec) +
                  (end.tv_usec - start.tv_usec) / 1000.0;
    printf("OpenMP Time: %.2f ms\n", time);

    return 0;
}
