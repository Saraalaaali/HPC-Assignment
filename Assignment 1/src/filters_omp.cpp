/*
 * filters_omp.cpp
 * ===============
 * OpenMP-parallel versions of all four image filters.
 *
 * The logic is identical to filters_serial.cpp.
 * The only addition is the OpenMP pragma that distributes
 * rows across available CPU threads:
 *
 *   #pragma omp parallel for schedule(static)
 *
 * "schedule(static)" splits the row range into equal chunks
 * and assigns one chunk per thread at compile time — lowest
 * overhead when every row takes roughly the same amount of work
 * (which is true for all our convolution filters).
 */

#include "filters.h"
#include <omp.h>
#include <cmath>
#include <algorithm>

// ============================================================
// Helper: clamp float to [0, 255]
// ============================================================
static inline float clampf(float val) {
    return std::max(0.0f, std::min(255.0f, val));
}

// ============================================================
// apply_kernel_omp
// ----------------
// Parallel convolution: same nested loop as the serial version,
// but the outer "row" loop is parallelised with OpenMP so each
// thread handles a distinct subset of image rows.
// ============================================================
static void apply_kernel_omp(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols,
    const std::vector<float>& kernel,
    int k,
    bool normalize,
    int num_threads)
{
    omp_set_num_threads(num_threads);

    int half = k / 2;
    float kernel_sum = 0.0f;
    for (float v : kernel) kernel_sum += v;
    if (kernel_sum == 0.0f) kernel_sum = 1.0f;

    // Each thread processes a contiguous band of rows.
    // There are no data dependencies between different rows,
    // so no locks or reductions are needed — just plain parallelism.
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            float sum = 0.0f;

            for (int ki = 0; ki < k; ki++) {
                for (int kj = 0; kj < k; kj++) {
                    int img_row = std::max(0, std::min(row + ki - half, rows - 1));
                    int img_col = std::max(0, std::min(col + kj - half, cols - 1));
                    sum += input[img_row * cols + img_col] * kernel[ki * k + kj];
                }
            }

            if (normalize) sum /= kernel_sum;
            output[row * cols + col] = sum;
        }
    }
}

// ============================================================
// Individual filter wrappers — build the kernel and delegate
// ============================================================

void box_blur_omp(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k, int num_threads)
{
    std::vector<float> kernel(k * k, 1.0f);  // uniform weights
    apply_kernel_omp(input, output, rows, cols, kernel, k, true, num_threads);
}

void gaussian_blur_omp(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k, int num_threads)
{
    float sigma = (k - 1) / 6.0f;
    if (sigma < 0.5f) sigma = 0.5f;

    std::vector<float> kernel(k * k);
    int half = k / 2;
    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            float x = i - half, y = j - half;
            kernel[i * k + j] = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
        }

    apply_kernel_omp(input, output, rows, cols, kernel, k, true, num_threads);
}

void sharpen_omp(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k, int num_threads)
{
    // Step 1: parallel Gaussian blur
    std::vector<float> blurred(rows * cols);
    gaussian_blur_omp(input, blurred, rows, cols, k, num_threads);

    // Step 2: parallel Unsharp Masking — no per-element dependencies
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < rows * cols; i++) {
        output[i] = clampf(input[i] + 1.5f * (input[i] - blurred[i]));
    }
}

void sobel_omp(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int /*k*/, int num_threads)
{
    // Fixed 3×3 Sobel kernels
    std::vector<float> Gx = {-1,0,1, -2,0,2, -1,0,1};
    std::vector<float> Gy = {-1,-2,-1, 0,0,0, 1,2,1};

    std::vector<float> grad_x(rows * cols);
    std::vector<float> grad_y(rows * cols);

    // Both convolutions run in parallel (each using all threads)
    apply_kernel_omp(input, grad_x, rows, cols, Gx, 3, false, num_threads);
    apply_kernel_omp(input, grad_y, rows, cols, Gy, 3, false, num_threads);

    // Combine gradients in parallel
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int i = 0; i < rows * cols; i++) {
        output[i] = std::min(255.0f,
                             std::sqrt(grad_x[i]*grad_x[i] + grad_y[i]*grad_y[i]));
    }
}

// ============================================================
// apply_filter_omp — public dispatcher
// ============================================================
void apply_filter_omp(FilterType ft,
                      const std::vector<float>& in,
                      std::vector<float>& out,
                      int rows, int cols, int k,
                      int num_threads)
{
    switch (ft) {
        case FilterType::BOX:      box_blur_omp (in, out, rows, cols, k, num_threads); break;
        case FilterType::GAUSSIAN: gaussian_blur_omp(in, out, rows, cols, k, num_threads); break;
        case FilterType::SHARPEN:  sharpen_omp  (in, out, rows, cols, k, num_threads); break;
        case FilterType::SOBEL:    sobel_omp    (in, out, rows, cols, k, num_threads); break;
    }
}
