#include "simd_kernel.h"
#include <immintrin.h>

void avx512_row_xor(uint64_t* matrix, int row_id, uint64_t* eliminator) {
    constexpr int step = sizeof(__m512i) / sizeof(uint64_t);
    const int total_blocks = matrix_cols / 64; // 64位块处理

#pragma omp simd aligned(matrix, eliminator: 64)
    for (int i = 0; i < total_blocks; i += step) {
        __m512i vec_row = _mm512_load_epi64(&matrix[row_id * total_blocks + i]);
        __m512i vec_ele = _mm512_load_epi64(&eliminator[i]);
        __m512i result = _mm512_xor_epi64(vec_row, vec_ele);
        _mm512_storeu_epi64(&matrix[row_id * total_blocks + i], result);
    }

    // 尾部处理
    int remainder = total_blocks % step;
    for (int i = total_blocks - remainder; i < total_blocks; i++) {
        matrix[row_id * total_blocks + i] ^= eliminator[i];
    }
}