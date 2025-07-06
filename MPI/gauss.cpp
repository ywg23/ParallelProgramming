#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string.h>
#include <omp.h>

using namespace std;

#define ElementType float
#define NEAR_ZERO 1e-5f

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 2048
#endif

#ifndef DATA_FILE
#define DATA_FILE "/home/suhipek/NKU_parallel_programming/gauss.dat"
#endif

int total_ranks, current_rank;

void masterProcess(ElementType *matrix_data) {
    ElementType(*matrix)[MATRIX_SIZE] = (ElementType(*)[MATRIX_SIZE])matrix_data;
    
    for (int pivot = 0; pivot < MATRIX_SIZE; pivot++) {
        int segment_size = (MATRIX_SIZE - pivot - 1 + total_ranks - 1) / total_ranks;
        segment_size = (segment_size > 1) ? segment_size : 0; // Skip if only one row per segment

        // Broadcast pivot row to all workers
        if (segment_size) {
            MPI_Bcast(matrix[pivot], MATRIX_SIZE * sizeof(ElementType), MPI_BYTE, 0, MPI_COMM_WORLD);
            
            // Distribute row segments to workers
            for (int worker = 1; worker < total_ranks; worker++) {
                int start_row = pivot + 1 + (worker - 1) * segment_size;
                if (start_row < MATRIX_SIZE) {
                    MPI_Send(matrix[start_row], segment_size * MATRIX_SIZE * sizeof(ElementType),
                             MPI_BYTE, worker, 0, MPI_COMM_WORLD);
                }
            }
        }

        // Process local rows (remaining after distribution)
        int local_start = pivot + 1 + (total_ranks - 1) * segment_size;
        #pragma omp parallel for num_threads(4)
        for (int row = local_start; row < MATRIX_SIZE; row++) {
            if (fabs(matrix[pivot][pivot]) < NEAR_ZERO) continue;
            
            ElementType factor = matrix[row][pivot] / matrix[pivot][pivot];
            #pragma omp simd
            for (int col = pivot; col < MATRIX_SIZE; col++) {
                matrix[row][col] -= matrix[pivot][col] * factor;
            }
        }

        // Collect processed segments from workers
        if (segment_size) {
            for (int worker = 1; worker < total_ranks; worker++) {
                int start_row = pivot + 1 + (worker - 1) * segment_size;
                if (start_row < MATRIX_SIZE) {
                    MPI_Recv(matrix[start_row], segment_size * MATRIX_SIZE * sizeof(ElementType),
                             MPI_BYTE, worker, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    }
}

void workerProcess() {
    ElementType pivot_row[MATRIX_SIZE]; // Stores broadcasted pivot row

    for (int pivot = 0; pivot < MATRIX_SIZE; pivot++) {
        int segment_size = (MATRIX_SIZE - pivot - 1 + total_ranks - 1) / total_ranks;
        if (segment_size <= 1) break; // No work for workers

        // Receive pivot row from master
        MPI_Bcast(pivot_row, MATRIX_SIZE * sizeof(ElementType), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Calculate assigned row segment
        int start_index = pivot + 1 + (current_rank - 1) * segment_size;
        int actual_lines = min(segment_size, MATRIX_SIZE - start_index);
        if (actual_lines <= 0) continue;

        // Receive and process assigned rows
        ElementType *local_rows = new ElementType[actual_lines * MATRIX_SIZE];
        MPI_Recv(local_rows, actual_lines * MATRIX_SIZE * sizeof(ElementType),
                 MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < actual_lines; i++) {
            if (fabs(pivot_row[pivot]) < NEAR_ZERO) continue;
            
            ElementType factor = local_rows[i * MATRIX_SIZE + pivot] / pivot_row[pivot];
            #pragma omp simd
            for (int col = pivot; col < MATRIX_SIZE; col++) {
                local_rows[i * MATRIX_SIZE + col] -= pivot_row[col] * factor;
            }
        }

        // Send processed rows back to master
        MPI_Send(local_rows, actual_lines * MATRIX_SIZE * sizeof(ElementType),
                 MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        delete[] local_rows;
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    if (current_rank == 0) {
        ElementType *matrix = new ElementType[MATRIX_SIZE * MATRIX_SIZE];
        ifstream data_file(DATA_FILE, ios::binary);
        data_file.read((char *)matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(ElementType));
        data_file.close();

        double start_time = MPI_Wtime();
        masterProcess(matrix);
        double elapsed = MPI_Wtime() - start_time;
        cout << fixed << setprecision(4) << elapsed * 1000 << " ms" << endl;

        delete[] matrix;
    } else {
        workerProcess();
    }

    MPI_Finalize();
    return 0;
}
