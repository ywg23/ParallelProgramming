#pragma once
#include <cstdint>

enum StorageType { DENSE, SPARSE };

struct HybridMatrix {
    StorageType storage_type;

    // �����干���ڴ�ռ�
    union {
        struct {
            uint64_t* data;
            int rows, cols;
        } dense;

        struct {
            int* row_ptr;
            int* col_idx;
            uint64_t* values;
            int nnz;
        } sparse;
    };

    // ͳһ�ӿ�
    void row_xor(int target_row, int source_row);
};