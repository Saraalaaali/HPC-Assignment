/*
 * image_io.cpp
 * ============
 * Image loading/saving (via OpenCV) and the timing utility.
 *
 * load_image_gray  — reads any image format, converts to grayscale float
 * save_image       — writes a grayscale float array back to disk as PNG
 * measure_time_ms  — runs a function N times, prints avg ± stdev, returns avg
 */

#include "filters.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <iostream>

// ============================================================
// load_image_gray
// ---------------
// Opens the file at `path`, converts it to grayscale, and returns
// a flat float array [0, 255].  Sets rows/cols to the image size.
// ============================================================
std::vector<float> load_image_gray(const std::string& path, int& rows, int& cols)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    if (img.empty())
        throw std::runtime_error("Cannot open image: " + path);

    rows = img.rows;
    cols = img.cols;

    // Convert from uint8 [0,255] to float [0,255]
    std::vector<float> data(rows * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i * cols + j] = static_cast<float>(img.at<uchar>(i, j));

    return data;
}

// ============================================================
// save_image
// ----------
// Clamps values to [0,255], converts to uint8, writes as PNG.
// ============================================================
void save_image(const std::string& path, const std::vector<float>& data, int rows, int cols)
{
    cv::Mat img(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            float val = std::max(0.0f, std::min(255.0f, data[i * cols + j]));
            img.at<uchar>(i, j) = static_cast<uchar>(val);
        }

    cv::imwrite(path, img);
}

// ============================================================
// measure_time_ms
// ---------------
// Runs `func` `repeats` times using a high-resolution wall clock.
// Prints avg and standard deviation to stderr, returns the average.
//
// We use std::chrono::high_resolution_clock which gives nanosecond
// resolution on modern Linux systems.
// ============================================================
double measure_time_ms(std::function<void()> func, int repeats)
{
    std::vector<double> times(repeats);

    for (int i = 0; i < repeats; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end   = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Average
    double avg = std::accumulate(times.begin(), times.end(), 0.0) / repeats;

    // Standard deviation
    double variance = 0.0;
    for (double t : times)
        variance += (t - avg) * (t - avg);
    double stdev = std::sqrt(variance / repeats);

    std::cerr << "    [timing] avg=" << avg << " ms  stdev=" << stdev
              << " ms  (n=" << repeats << ")\n";

    return avg;
}

// ============================================================
// apply_filter_serial — dispatcher (defined here to avoid a
// separate compilation unit just for one switch statement)
// ============================================================
void apply_filter_serial(FilterType ft,
                         const std::vector<float>& in,
                         std::vector<float>& out,
                         int rows, int cols, int k)
{
    switch (ft) {
        case FilterType::BOX:      box_blur_serial      (in, out, rows, cols, k); break;
        case FilterType::GAUSSIAN: gaussian_blur_serial (in, out, rows, cols, k); break;
        case FilterType::SHARPEN:  sharpen_serial       (in, out, rows, cols, k); break;
        case FilterType::SOBEL:    sobel_serial         (in, out, rows, cols, k); break;
    }
}
