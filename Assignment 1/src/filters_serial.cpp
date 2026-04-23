/*
 * filters_serial.cpp
 * ==================
 * Serial (single-threaded CPU) implementation of all image filters.
 * This is the baseline we compare OpenMP and CUDA against.
 * Every pixel is processed one at a time, in order.
 */

#include "filters.h"
#include <cmath>
#include <stdexcept>

// ============================================================
// Helper: clamp a value to the valid pixel range [0, 255]
// ============================================================
static inline int clamp(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return val;
}

// ============================================================
// apply_kernel_serial
// -------------------
// Core convolution routine: slides a kernel over every pixel.
//
// Parameters:
//   input     - flat array of grayscale pixel values [0..255]
//   output    - result is written here (same size as input)
//   rows/cols - image dimensions
//   kernel    - square filter matrix (stored row-major, size k*k)
//   k         - kernel side length (must be odd: 3, 5, 7, 11, ...)
//   normalize - if true, divide by kernel sum (needed for blur filters
//               so the output brightness stays the same as the input)
//
// Border strategy: "clamp to edge" — pixels outside the image
// are replaced by the nearest border pixel.
// ============================================================
void apply_kernel_serial(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols,
    const std::vector<float>& kernel,
    int k,
    bool normalize)
{
    int half = k / 2;  // half-width of the kernel (e.g. k=7 → half=3)

    // Pre-compute kernel sum for normalization
    float kernel_sum = 0.0f;
    for (float v : kernel) kernel_sum += v;
    if (kernel_sum == 0.0f) kernel_sum = 1.0f;  // avoid division by zero

    // Iterate over every output pixel
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            float sum = 0.0f;

            // Apply the kernel: loop over each kernel element
            for (int ki = 0; ki < k; ki++) {
                for (int kj = 0; kj < k; kj++) {

                    // Map kernel position to image coordinates
                    int img_row = row + ki - half;
                    int img_col = col + kj - half;

                    // Clamp to edge if we go outside image bounds
                    img_row = std::max(0, std::min(img_row, rows - 1));
                    img_col = std::max(0, std::min(img_col, cols - 1));

                    float pixel  = input[img_row * cols + img_col];
                    float weight = kernel[ki * k + kj];
                    sum += pixel * weight;
                }
            }

            if (normalize) sum /= kernel_sum;

            output[row * cols + col] = sum;
        }
    }
}

// ============================================================
// box_blur_serial
// ---------------
// Simplest blur: all kernel weights = 1 (plain average of neighbours).
// Fast but produces a "blocky" blur compared to Gaussian.
// ============================================================
void box_blur_serial(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k)
{
    std::vector<float> kernel(k * k, 1.0f);  // all-ones kernel
    apply_kernel_serial(input, output, rows, cols, kernel, k, true);
}

// ============================================================
// gaussian_blur_serial
// --------------------
// Weighted blur: pixels close to the centre contribute more.
// The weights follow the 2-D Gaussian bell-curve formula:
//   w(x,y) = exp( -(x² + y²) / (2σ²) )
// This produces a smooth, natural-looking blur.
// ============================================================
void gaussian_blur_serial(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k)
{
    // Derive sigma from kernel size (common rule of thumb)
    float sigma = (k - 1) / 6.0f;
    if (sigma < 0.5f) sigma = 0.5f;

    std::vector<float> kernel(k * k);
    int half = k / 2;

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            float x = i - half;
            float y = j - half;
            kernel[i * k + j] = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
        }
    }

    apply_kernel_serial(input, output, rows, cols, kernel, k, true);
}

// ============================================================
// sharpen_serial
// --------------
// Makes edges look crisper using Unsharp Masking:
//   output = input + alpha * (input - blurred)
//
// Intuition: "blurred" is a low-frequency version of the image.
// Subtracting it from the original leaves only high-frequency
// detail (edges). Adding that detail back amplifies the edges.
// ============================================================
void sharpen_serial(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k)
{
    // Step 1: compute Gaussian blur
    std::vector<float> blurred(rows * cols);
    gaussian_blur_serial(input, blurred, rows, cols, k);

    // Step 2: Unsharp Masking
    float alpha = 1.5f;  // sharpening strength
    for (int i = 0; i < rows * cols; i++) {
        output[i] = input[i] + alpha * (input[i] - blurred[i]);
        output[i] = std::max(0.0f, std::min(255.0f, output[i]));  // clamp
    }
}

// ============================================================
// sobel_serial
// ------------
// Edge detection using the Sobel operator.
// Uses two 3×3 kernels to measure the image gradient:
//   Gx — detects horizontal edges (left ↔ right changes)
//   Gy — detects vertical edges   (top  ↕ bottom changes)
// Final magnitude = sqrt(Gx² + Gy²)
// ============================================================
void sobel_serial(
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int /*k — Sobel always uses a fixed 3×3 kernel*/)
{
    // Fixed 3×3 Sobel kernels
    std::vector<float> Gx = {
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    };
    std::vector<float> Gy = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };

    std::vector<float> grad_x(rows * cols);
    std::vector<float> grad_y(rows * cols);

    apply_kernel_serial(input, grad_x, rows, cols, Gx, 3, false);
    apply_kernel_serial(input, grad_y, rows, cols, Gy, 3, false);

    // Combine the two gradients into a single edge magnitude
    for (int i = 0; i < rows * cols; i++) {
        output[i] = std::sqrt(grad_x[i]*grad_x[i] + grad_y[i]*grad_y[i]);
        output[i] = std::min(255.0f, output[i]);
    }
}
