#include <mpi.h>
#include "parallel_scheduler.h"

void run_mpi_scheduler(HybridMatrix& matrix) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { // Master
        std::vector<MPI_Request> requests(size - 1);
        for (int worker = 1; worker < size; worker++) {
            MatrixTask task = get_next_task();
            MPI_Isend(&task, sizeof(MatrixTask), MPI_BYTE,
                worker, 0, MPI_COMM_WORLD, &requests[worker - 1]);
        }

        // 结果收集
        MPI_Waitall(size - 1, requests.data(), MPI_STATUSES_IGNORE);
    }
    else { // Worker
        MatrixTask task;
        MPI_Request recv_req;
        MPI_Irecv(&task, sizeof(MatrixTask), MPI_BYTE,
            0, 0, MPI_COMM_WORLD, &recv_req);

        // 异步准备GPU计算
        cudaEvent_t ready_event;
        cudaEventCreate(&ready_event);

        // 重叠通信和计算
        int recv_done;
        MPI_Test(&recv_req, &recv_done, MPI_STATUS_IGNORE);
        if (recv_done) {
            launch_gpu_computation(task);
            cudaEventRecord(ready_event);
        }

        // 结果返回
        MPI_Send(result_buf, result_size, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
    }
}
}