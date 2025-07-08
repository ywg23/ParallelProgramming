#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define MAX_N 1024
#define NUM_THREADS 8

float A[MAX_N][MAX_N];
int n = 1024;  // 矩阵规模

pthread_barrier_t barrier_division;
pthread_barrier_t barrier_elimination;

typedef struct {
    int t_id;
} threadParam_t;

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

void* threadFunc(void* param) {
    int t_id = ((threadParam_t*)param)->t_id;
    for (int k = 0; k < n; ++k) {
        if (t_id == 0) {
            float pivot = A[k][k];
            for (int j = k + 1; j < n; ++j)
                A[k][j] /= pivot;
            A[k][k] = 1.0;
        }

        pthread_barrier_wait(&barrier_division);

        for (int i = k + 1 + t_id; i < n; i += NUM_THREADS) {
            float factor = A[i][k];
            for (int j = k + 1; j < n; ++j)
                A[i][j] -= factor * A[k][j];
            A[i][k] = 0.0;
        }

        pthread_barrier_wait(&barrier_elimination);
    }

    pthread_exit(NULL);
}

int main() {
    init_matrix();

    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    pthread_barrier_init(&barrier_division, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elimination, NULL, NUM_THREADS);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, &param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    gettimeofday(&end, NULL);
    double time = 1000.0 * (end.tv_sec - start.tv_sec) +
                  (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Pthread Time: %.2f ms\n", time);

    pthread_barrier_destroy(&barrier_division);
    pthread_barrier_destroy(&barrier_elimination);

    return 0;
}
