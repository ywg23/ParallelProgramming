#include "parallel_scheduler.h"
#include <queue>
#include <mutex>
#include <omp.h>

class TaskPool {
    std::queue<MatrixTask> taskQueue;
    std::mutex queueMutex;

public:
    void addTask(const MatrixTask& task) {
        std::lock_guard<std::mutex> lock(queueMutex);
        taskQueue.push(task);
    }

    bool getTask(MatrixTask& task) {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (taskQueue.empty()) return false;
        task = taskQueue.front();
        taskQueue.pop();
        return true;
    }
};

void run_parallel_scheduler(HybridMatrix& matrix) {
    TaskPool globalPool;
    initialize_tasks(globalPool, matrix); // 任务初始化

#pragma omp parallel num_threads(32)
    {
        MatrixTask task;
        while (true) {
            if (globalPool.getTask(task)) {
                process_matrix_task(task);
            }
            else if (omp_get_thread_num() % 5 == 0) { // 20%概率窃取
                steal_work_from_neighbor();
            }
            else {
                break;
            }
        }
    }
}