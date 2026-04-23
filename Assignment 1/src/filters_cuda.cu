/*
 * filters_cuda.cu
 * ===============
 * GPU (CUDA) implementations of all four image filters.
 *
 * Key idea — Shared Memory Tiling
 * --------------------------------
 * Global GPU memory is slow (~400 cycles per access).
 * Shared memory is fast (~4 cycles), but small and scoped to one block.
 *
 * Strategy:
 *   1. Divide the image into rectangular "tiles", one tile per block.
 *   2. Every thread in the block cooperates to load its tile — plus
 *      a surrounding "halo" of pixels needed by border threads — into
 *      shared memory.
 *   3. All threads then read from shared memory for the convolution.
 *
 * What is the Halo?
 *   A thread on the edge of a tile needs pixels from outside its tile
 *   to apply the kernel.  Those extra pixels (half = k/2 wide on each
 *   side) are the "halo".  The shared-memory tile is therefore:
 *
 *       tile_width  = block_width  + 2 * half
 *       tile_height = block_height + 2 * half
 *
 *   Diagram (k=3, half=1, block 4x4):
 *
 *       H H H H H H   <- halo row (top)
 *       H * * * * H
 *       H * * * * H   * = pixels this block "owns"
 *       H * * * * H   H = halo pixels (loaded but not written as output)
 *       H * * * * H
 *       H H H H H H   <- halo row (bottom)
 */

#include "filters.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

// ============================================================
// CUDA error-checking macro
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("CUDA Error: ") + cudaGetErrorString(err) +    \
                " at line " + std::to_string(__LINE__));                    \
        }                                                                   \
    } while(0)

// ============================================================
// apply_kernel_gpu
// ----------------
// General-purpose convolution kernel using shared memory tiling.
//
// Grid/Block layout:
//   gridDim.x  = ceil(cols / blockDim.x)
//   gridDim.y  = ceil(rows / blockDim.y)
//   Each thread computes exactly one output pixel.
// ============================================================
__global__ void apply_kernel_gpu(
    const float* __restrict__ input,   // image in GPU global memory
    float*       output,               // result in GPU global memory
    int rows, int cols,
    const float* __restrict__ kernel,  // filter kernel in GPU memory
    int k,
    float kernel_sum,
    bool normalize)
{
    int half = k / 2;

    // Shared memory tile — sized for worst-case kernel (k=11, half=5)
    // with max block size 32x32
    const int TILE_W = 32 + 2 * 5;  // 42
    const int TILE_H = 32 + 2 * 5;  // 42

    __shared__ float tile[TILE_H][TILE_W];

    // Global pixel coordinates this thread will write
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Top-left corner of this tile in the full image (including halo offset)
    int tile_start_col = blockIdx.x * blockDim.x - half;
    int tile_start_row = blockIdx.y * blockDim.y - half;

    // Actual tile dimensions = block + halo on both sides
    int tile_w = blockDim.x + 2 * half;
    int tile_h = blockDim.y + 2 * half;

    // Cooperative loading: stride by blockDim so every element gets loaded
    for (int ty = threadIdx.y; ty < tile_h; ty += blockDim.y) {
        for (int tx = threadIdx.x; tx < tile_w; tx += blockDim.x) {
            int ir = max(0, min(tile_start_row + ty, rows - 1));  // clamp to edge
            int ic = max(0, min(tile_start_col + tx, cols - 1));
            tile[ty][tx] = input[ir * cols + ic];
        }
    }

    // Wait until ALL threads in this block finish loading before any reads
    __syncthreads();

    // Threads outside image bounds write nothing
    if (out_row >= rows || out_col >= cols) return;

    // Convolve from fast shared memory
    float sum = 0.0f;
    for (int ki = 0; ki < k; ki++)
        for (int kj = 0; kj < k; kj++)
            sum += tile[threadIdx.y + ki][threadIdx.x + kj] * kernel[ki * k + kj];

    if (normalize) sum /= kernel_sum;
    output[out_row * cols + out_col] = sum;
}

// ============================================================
// sobel_kernel_gpu
// ----------------
// Dedicated Sobel kernel: computes Gx and Gy in a single pass.
// Fixed 3x3 kernel means halo is always exactly 1 pixel wide.
// ============================================================
__global__ void sobel_kernel_gpu(
    const float* __restrict__ input,
    float* output,
    int rows, int cols)
{
    // Sobel kernels stored in registers (fastest possible access)
    const float Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = {{-1,-2,-1}, { 0, 0, 0}, { 1, 2, 1}};

    // Tile = block (max 32x32) + 1-pixel halo on each side = 34x34
    __shared__ float tile[34][34];

    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    int tile_start_col = blockIdx.x * blockDim.x - 1;
    int tile_start_row = blockIdx.y * blockDim.y - 1;

    // Load (blockDim+2) x (blockDim+2) tile
    for (int ty = threadIdx.y; ty < (int)blockDim.y + 2; ty += blockDim.y)
        for (int tx = threadIdx.x; tx < (int)blockDim.x + 2; tx += blockDim.x) {
            int ir = max(0, min(tile_start_row + ty, rows - 1));
            int ic = max(0, min(tile_start_col + tx, cols - 1));
            tile[ty][tx] = input[ir * cols + ic];
        }
    __syncthreads();

    if (out_row >= rows || out_col >= cols) return;

    // Compute both gradients in one pass (avoids launching two kernels)
    float gx = 0.0f, gy = 0.0f;
    for (int ki = 0; ki < 3; ki++)
        for (int kj = 0; kj < 3; kj++) {
            float p = tile[threadIdx.y + ki][threadIdx.x + kj];
            gx += p * Gx[ki][kj];
            gy += p * Gy[ki][kj];
        }

    output[out_row * cols + out_col] = fminf(255.0f, sqrtf(gx*gx + gy*gy));
}

// ============================================================
// Helper: copy a kernel vector from CPU to GPU memory
// ============================================================
static float* upload_kernel_to_gpu(const std::vector<float>& kernel) {
    float* d_kernel;
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data(),
                          kernel.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return d_kernel;
}

// ============================================================
// run_filter_cuda
// ---------------
// Full GPU pipeline:
//   [CPU] upload image → [GPU] run kernel → [CPU] download result
// ============================================================
static void run_filter_cuda(
    FilterType ft,
    const std::vector<float>& input,
    std::vector<float>& output,
    int rows, int cols, int k,
    int block_x, int block_y)
{
    int N = rows * cols;

    // Allocate GPU buffers
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // Upload image to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                          N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure launch geometry:
    //   block = block_x * block_y threads
    //   grid  = enough blocks to cover the whole image
    dim3 block(block_x, block_y);
    dim3 grid(
        (cols + block_x - 1) / block_x,
        (rows + block_y - 1) / block_y
    );

    if (ft == FilterType::SOBEL) {
        sobel_kernel_gpu<<<grid, block>>>(d_input, d_output, rows, cols);

    } else {
        // Build filter kernel on CPU, then upload
        std::vector<float> kernel;

        if (ft == FilterType::BOX) {
            kernel.assign(k * k, 1.0f);
        } else {
            // Gaussian weights (shared by GAUSSIAN and SHARPEN)
            float sigma = (k - 1) / 6.0f;
            if (sigma < 0.5f) sigma = 0.5f;
            kernel.resize(k * k);
            int half = k / 2;
            for (int i = 0; i < k; i++)
                for (int j = 0; j < k; j++) {
                    float x = i - half, y = j - half;
                    kernel[i*k+j] = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
                }
        }

        float ksum = 0.0f;
        for (float v : kernel) ksum += v;

        float* d_kernel = upload_kernel_to_gpu(kernel);

        if (ft == FilterType::SHARPEN) {
            // Step 1: Gaussian blur on the GPU
            float* d_blurred;
            CUDA_CHECK(cudaMalloc(&d_blurred, N * sizeof(float)));
            apply_kernel_gpu<<<grid, block>>>(
                d_input, d_blurred, rows, cols, d_kernel, k, ksum, true);
            CUDA_CHECK(cudaGetLastError());

            // Step 2: Unsharp Masking on the CPU
            // (keeps code simple; the blur dominates runtime anyway)
            std::vector<float> blurred_host(N);
            CUDA_CHECK(cudaMemcpy(blurred_host.data(), d_blurred,
                                  N * sizeof(float), cudaMemcpyDeviceToHost));

            std::vector<float> sharp(N);
            for (int i = 0; i < N; i++)
                sharp[i] = std::max(0.0f, std::min(255.0f,
                    input[i] + 1.5f * (input[i] - blurred_host[i])));

            CUDA_CHECK(cudaMemcpy(d_output, sharp.data(),
                                  N * sizeof(float), cudaMemcpyHostToDevice));
            cudaFree(d_blurred);

        } else {
            // Box or Gaussian: single-pass convolution
            apply_kernel_gpu<<<grid, block>>>(
                d_input, d_output, rows, cols, d_kernel, k, ksum, true);
            CUDA_CHECK(cudaGetLastError());
        }

        cudaFree(d_kernel);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Download result to CPU
    output.resize(N);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                          N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}

// ============================================================
// apply_filter_cuda — public entry point
// ============================================================
void apply_filter_cuda(FilterType ft,
                       const std::vector<float>& in,
                       std::vector<float>& out,
                       int rows, int cols, int k,
                       int block_dim_x, int block_dim_y)
{
    run_filter_cuda(ft, in, out, rows, cols, k, block_dim_x, block_dim_y);
}
