/*
 * filters.h
 * =========
 * Central header: declares every function used across the project.
 * All .cpp and .cu files include this.
 */

#pragma once
#include <vector>
#include <string>
#include <functional>

// ============================================================
// Filter type enum — passed around instead of raw strings
// ============================================================
enum class FilterType {
    BOX,       // plain average blur
    GAUSSIAN,  // weighted bell-curve blur
    SHARPEN,   // unsharp masking
    SOBEL      // edge detection
};

// ============================================================
// Serial implementations  (filters_serial.cpp)
// ============================================================
void apply_kernel_serial(const std::vector<float>& in, std::vector<float>& out,
                         int rows, int cols,
                         const std::vector<float>& kernel, int k, bool normalize);

void box_blur_serial      (const std::vector<float>& in, std::vector<float>& out, int rows, int cols, int k);
void gaussian_blur_serial (const std::vector<float>& in, std::vector<float>& out, int rows, int cols, int k);
void sharpen_serial       (const std::vector<float>& in, std::vector<float>& out, int rows, int cols, int k);
void sobel_serial         (const std::vector<float>& in, std::vector<float>& out, int rows, int cols, int k);

// Dispatcher: picks the right serial function based on the enum
void apply_filter_serial(FilterType ft,
                         const std::vector<float>& in,
                         std::vector<float>& out,
                         int rows, int cols, int k);

// ============================================================
// OpenMP implementations  (filters_omp.cpp)
// ============================================================
void apply_filter_omp(FilterType ft,
                      const std::vector<float>& in,
                      std::vector<float>& out,
                      int rows, int cols, int k,
                      int num_threads);

// ============================================================
// CUDA implementations  (filters_cuda.cu)
// ============================================================
void apply_filter_cuda(FilterType ft,
                       const std::vector<float>& in,
                       std::vector<float>& out,
                       int rows, int cols, int k,
                       int block_dim_x, int block_dim_y);

// ============================================================
// Image I/O  (image_io.cpp)
// ============================================================

// Load any image from disk and convert it to grayscale float [0, 255].
// rows and cols are set to the image dimensions on return.
std::vector<float> load_image_gray(const std::string& path, int& rows, int& cols);

// Save a grayscale float [0, 255] array as a PNG/JPG file.
void save_image(const std::string& path, const std::vector<float>& data, int rows, int cols);

// ============================================================
// Timing utility  (image_io.cpp)
// ============================================================

// Run `func` `repeats` times, print avg ± stdev, and return the average in ms.
double measure_time_ms(std::function<void()> func, int repeats = 5);
