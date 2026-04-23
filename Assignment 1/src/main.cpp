#include "filters.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <stdexcept>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;

// ============================================================
// دالة التحقق العددي: حساب متوسط مربع الخطأ (MSE)
// ============================================================
double calculate_mse(const std::vector<float>& img1, const std::vector<float>& img2) {
    if (img1.size() != img2.size()) return -1.0;
    double mse = 0;
    for (size_t i = 0; i < img1.size(); i++) {
        double diff = img1[i] - img2[i];
        mse += diff * diff;
    }
    return mse / img1.size();
}

struct Config {
    std::string image_path;
    std::string filter_name = "gaussian";
    std::string impl = "serial";
    int kernel_size = 3;
    int threads = 4;
    int block_x = 16;
    int block_y = 16;
    int repeats = 5;
    std::string output_dir = "results";
    std::string csv_path = "results/results.csv";
};

void print_system_info() {
    std::cout << "\n=== System Info ===\n"
              << "CPU: Intel(R) Core(TM) i5-10500H CPU @ 2.50GHz\n"
              << "Cores: 12\n"
              << "OS: Linux 6.6.87.2-microsoft-standard-WSL2 x86_64\n"
              << "NVIDIA GeForce GTX 1650, 4096 MiB, 7.5\n"
              << "===================\n\n";
}

void append_csv(const std::string& path, const std::string& filter, const std::string& impl,
                int r, int c, int k, int t, int bx, int by, double time, double mse) {
    bool exists = fs::exists(path);
    std::ofstream ofs(path, std::ios::app);
    if (!exists) {
        ofs << "filter,impl,rows,cols,kernel,threads,block_x,block_y,time_ms,mse\n";
    }
    ofs << filter << "," << impl << "," << r << "," << c << "," << k << ","
        << t << "," << bx << "," << by << "," << time << "," << mse << "\n";
}

int main(int argc, char** argv) {
    try {
        Config cfg;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--image") cfg.image_path = argv[++i];
            else if (arg == "--filter") cfg.filter_name = argv[++i];
            else if (arg == "--impl") cfg.impl = argv[++i];
            else if (arg == "--kernel") cfg.kernel_size = std::stoi(argv[++i]);
            else if (arg == "--threads") cfg.threads = std::stoi(argv[++i]);
            else if (arg == "--block-x") cfg.block_x = std::stoi(argv[++i]);
            else if (arg == "--block-y") cfg.block_y = std::stoi(argv[++i]);
            else if (arg == "--repeats") cfg.repeats = std::stoi(argv[++i]);
            else if (arg == "--output") cfg.output_dir = argv[++i];
            else if (arg == "--csv") cfg.csv_path = argv[++i];
        }

        if (cfg.image_path.empty()) return 1;

        print_system_info();

        int rows, cols;
        std::vector<float> image = load_image_gray(cfg.image_path, rows, cols);
        std::vector<float> output(rows * cols);
        
        std::cout << "Loading image: " << cfg.image_path << "\n";
        std::cout << "  Size: " << cols << " x " << rows << " pixels\n\n";

        FilterType ft;
        if (cfg.filter_name == "box") ft = FilterType::BOX;
        else if (cfg.filter_name == "gaussian") ft = FilterType::GAUSSIAN;
        else if (cfg.filter_name == "sharpen") ft = FilterType::SHARPEN;
        else if (cfg.filter_name == "sobel") ft = FilterType::SOBEL;
        else throw std::runtime_error("Unknown filter: " + cfg.filter_name);

        std::cout << "Filter : " << cfg.filter_name << "  |  Impl: " << cfg.impl 
                  << "  |  Kernel: " << cfg.kernel_size << "x" << cfg.kernel_size;
        if (cfg.impl == "omp") std::cout << "  |  Threads: " << cfg.threads;
        if (cfg.impl == "cuda") std::cout << "  |  Block: " << cfg.block_x << "x" << cfg.block_y;
        std::cout << "\n";

        double time_ms = 0;
        double mse = 0.0;

        // تنفيذ الحسابات وقياس الوقت
        if (cfg.impl == "serial") {
            time_ms = measure_time_ms([&]() {
                apply_filter_serial(ft, image, output, rows, cols, cfg.kernel_size);
            }, cfg.repeats);
        } else if (cfg.impl == "omp") {
            time_ms = measure_time_ms([&]() {
                apply_filter_omp(ft, image, output, rows, cols, cfg.kernel_size, cfg.threads);
            }, cfg.repeats);
        } else if (cfg.impl == "cuda") {
            time_ms = measure_time_ms([&]() {
                apply_filter_cuda(ft, image, output, rows, cols, cfg.kernel_size, cfg.block_x, cfg.block_y);
            }, cfg.repeats);
        }

        // جزء الـ Numeric Check
        if (cfg.impl != "serial") {
            std::vector<float> serial_ref(rows * cols);
            apply_filter_serial(ft, image, serial_ref, rows, cols, cfg.kernel_size);
            mse = calculate_mse(output, serial_ref);
            
            std::cout << "Validation (vs Serial): ";
            if (mse < 1e-4) std::cout << "PASSED (MSE: " << mse << ")\n";
            else std::cout << "FAILED (MSE: " << mse << ")\n";
        }

        std::cout << "Time: " << time_ms << " ms\n";

        // حفظ الصورة والـ CSV
        fs::create_directories(cfg.output_dir);
        std::string out_img = cfg.output_dir + "/" + cfg.filter_name + "_" + cfg.impl + "_k" + std::to_string(cfg.kernel_size) + ".png";
        save_image(out_img, output, rows, cols);
        std::cout << "Saved: " << out_img << "\n";
        std::cout << "CSV:   " << cfg.csv_path << "\n";

        append_csv(cfg.csv_path, cfg.filter_name, cfg.impl, rows, cols, cfg.kernel_size, cfg.threads, cfg.block_x, cfg.block_y, time_ms, mse);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}