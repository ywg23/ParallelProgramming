#include <immintrin.h>
#include <emmintrin.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstring>
#include <cpuid.h>

// CPU feature detection
bool cpu_supports_sse41() {
    unsigned int eax, ebx, ecx, edx;
    return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (ecx & (1 << 19));
}

bool cpu_supports_avx() {
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    bool has_avx = ecx & (1 << 28);
    bool osxsave = ecx & (1 << 27);
    if (!(has_avx && osxsave)) return false;
    unsigned int a, d;
    __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    return (a & 0x6) == 0x6;
}

bool cpu_supports_avx2() {
    unsigned int eax, ebx, ecx, edx;
    return __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1 << 5));
}

// Basic Gaussian elimination (scalar)
void gaussian_elimination_normal(float* A, int n) {
    for (int k = 0; k < n; ++k) {
        float pivot = A[k * n + k];
        for (int j = k + 1; j < n; ++j) A[k * n + j] /= pivot;
        A[k * n + k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            float factor = A[i * n + k];
            for (int j = k + 1; j < n; ++j) A[i * n + j] -= factor * A[k * n + j];
            A[i * n + k] = 0.0f;
        }
    }
}

// SSE version: 4-way vectorization
void gaussian_elimination_sse(float* A, int n) {
    for (int k = 0; k < n; ++k) {
        __m128 vt = _mm_set1_ps(A[k * n + k]);
        int j = k + 1;
        for (; j + 4 <= n; j += 4) {
            __m128 va = _mm_loadu_ps(A + k * n + j);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(A + k * n + j, va);
        }
        for (; j < n; ++j) A[k * n + j] /= A[k * n + k];
        A[k * n + k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            __m128 vaik = _mm_set1_ps(A[i * n + k]);
            int jj = k + 1;
            for (; jj + 4 <= n; jj += 4) {
                __m128 vakj = _mm_loadu_ps(A + k * n + jj);
                __m128 vaij = _mm_loadu_ps(A + i * n + jj);
                vaij = _mm_sub_ps(vaij, _mm_mul_ps(vakj, vaik));
                _mm_storeu_ps(A + i * n + jj, vaij);
            }
            for (; jj < n; ++jj) A[i * n + jj] -= A[k * n + jj] * A[i * n + k];
            A[i * n + k] = 0.0f;
        }
    }
}

// AVX version: 8-way vectorization
void gaussian_elimination_avx(float* A, int n) {
    for (int k = 0; k < n; ++k) {
        __m256 vt = _mm256_set1_ps(A[k * n + k]);
        int j = k + 1;
        for (; j + 8 <= n; j += 8) {
            __m256 va = _mm256_loadu_ps(A + k * n + j);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(A + k * n + j, va);
        }
        for (; j < n; ++j) A[k * n + j] /= A[k * n + k];
        A[k * n + k] = 1.0f;
        for (int i = k + 1; i < n; ++i) {
            __m256 vaik = _mm256_set1_ps(A[i * n + k]);
            int jj = k + 1;
            for (; jj + 8 <= n; jj += 8) {
                __m256 vakj = _mm256_loadu_ps(A + k * n + jj);
                __m256 vaij = _mm256_loadu_ps(A + i * n + jj);
                vaij = _mm256_sub_ps(vaij, _mm256_mul_ps(vakj, vaik));
                _mm256_storeu_ps(A + i * n + jj, vaij);
            }
            for (; jj < n; ++jj) A[i * n + jj] -= A[k * n + jj] * A[i * n + k];
            A[i * n + k] = 0.0f;
        }
    }
}

// Utility routines
void generate_random_matrix(float* A, int n) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    for (int i = 0; i < n * n; ++i) A[i] = dist(rng);
}

void copy_matrix(float* dst, const float* src, int n) {
    std::memcpy(dst, src, sizeof(float) * n * n);
}

int main() {
    const int n = 1984;
    const int REPEAT = 30;

    // Allocate aligned matrices
    float* base = (float*)_mm_malloc(sizeof(float) * n * n, 32);
    float* A0 = (float*)_mm_malloc(sizeof(float) * n * n, 32);
    float* A1 = (float*)_mm_malloc(sizeof(float) * n * n, 32);
    float* A2 = (float*)_mm_malloc(sizeof(float) * n * n, 32);

    generate_random_matrix(base, n);

    double sum_norm = 0, sum_sse = 0, sum_avx = 0;

    for (int r = 0; r < REPEAT; ++r) {
        // Scalar
        copy_matrix(A0, base, n);
        auto t0 = std::chrono::high_resolution_clock::now();
        gaussian_elimination_normal(A0, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        sum_norm += std::chrono::duration<double>(t1 - t0).count();

        // SSE
        if (cpu_supports_sse41()) {
            copy_matrix(A1, base, n);
            auto ts = std::chrono::high_resolution_clock::now();
            gaussian_elimination_sse(A1, n);
            auto te = std::chrono::high_resolution_clock::now();
            sum_sse += std::chrono::duration<double>(te - ts).count();
        }

        // AVX
        if (cpu_supports_avx2() && cpu_supports_avx()) {
            copy_matrix(A2, base, n);
            auto ta = std::chrono::high_resolution_clock::now();
            gaussian_elimination_avx(A2, n);
            auto tb = std::chrono::high_resolution_clock::now();
            sum_avx += std::chrono::duration<double>(tb - ta).count();
        }
    }

    double avg_norm = sum_norm / REPEAT;
    double avg_sse = sum_sse / (cpu_supports_sse41() ? REPEAT : 1);
    double avg_avx = sum_avx / ((cpu_supports_avx2() && cpu_supports_avx()) ? REPEAT : 1);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average Scalar time: " << avg_norm << " s\n";
    if (cpu_supports_sse41())
        std::cout << "Average SSE time:    " << avg_sse << " s, speedup=" << avg_norm / avg_sse << "x\n";
    else
        std::cout << "SSE not supported\n";
    if (cpu_supports_avx2() && cpu_supports_avx())
        std::cout << "Average AVX time:    " << avg_avx << " s, speedup=" << avg_norm / avg_avx << "x\n";
    else
        std::cout << "AVX not supported\n";

    _mm_free(base);
    _mm_free(A0);
    _mm_free(A1);
    _mm_free(A2);
    return 0;
}
