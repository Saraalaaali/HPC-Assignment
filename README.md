# Project 3 — Image Filtering Pipeline
## شرح المشروع والتشغيل

---

## 📁 هيكل المشروع

```
project3_image_filter/
├── src/
│   ├── filters.h           # تعريف كل الدوال (header)
│   ├── filters_serial.cpp  # الـ filters بدون parallel
│   ├── filters_omp.cpp     # الـ filters بـ OpenMP
│   ├── filters_cuda.cu     # الـ filters بـ CUDA
│   ├── image_io.cpp        # تحميل وحفظ الصور + قياس الوقت
│   └── main.cpp            # البرنامج الرئيسي + CLI
├── CMakeLists.txt          # ملف البناء
├── run_experiments.sh      # سكريبت التجارب
└── README.md               # الملف ده
└── test images (input.jpg==>1920*1080 , input2.jpg==>4k)               

```

---

## 🔧 المتطلبات

```bash
# على Ubuntu:
sudo apt update
sudo apt install -y g++ cmake make
sudo apt install -y libopencv-dev     # OpenCV
sudo apt install -y libomp-dev        # OpenMP

# CUDA Toolkit (من موقع NVIDIA)
# https://developer.nvidia.com/cuda-downloads
```

---

## 🏗️ البناء (Build)

```bash
# من مجلد المشروع:
mkdir build
cd build
cmake ..
make -j$(nproc)   # استخدم كل الـ cores المتاحة
cd ..
```

> **مهم:** قبل `cmake ..`، تأكد إن `CMAKE_CUDA_ARCHITECTURES` في `CMakeLists.txt`
> بيتطابق مع GPU-ك. شغّل `nvidia-smi` تعرف الـ architecture.

---

## 🚀 طريقة الاستخدام

### تشغيل واحد:
```bash
# Serial (4k image)
./build/image_filter --image input2.jpg --filter gaussian --kernel 7 --impl serial

# OpenMP بـ 8 threads (1920*1080 image)
./build/image_filter --image input.jpg --filter sobel --impl omp --threads 8

# CUDA بـ block 16x16
./build/image_filter --image photo.jpg --filter box --kernel 11 --impl cuda --block-x 16 --block-y 16
```

### كل التجارب مع بعض:
```bash
chmod +x run_experiments.sh
./run_experiments.sh --image photo.jpg
```

---

## 🎛️ الـ CLI Options

| Flag | القيم | الوصف |
|------|-------|-------|
| `--image` | path | مسار الصورة |
| `--filter` | box, gaussian, sharpen, sobel | نوع الـ filter |
| `--impl` | serial, omp, cuda | طريقة التنفيذ |
| `--kernel` | 3, 7, 11, ... | حجم الـ kernel (فردي) |
| `--threads` | 1, 2, 4, 8, 16 | عدد الـ OpenMP threads |
| `--block-x` | 16, 32 | حجم الـ CUDA block (x) |
| `--block-y` | 8, 16, 32 | حجم الـ CUDA block (y) |
| `--repeats` | 5 | عدد مرات التكرار |
| `--output` | path | مجلد النتائج |
| `--csv` | path | ملف CSV |

---

## 📊 الـ Filters المتاحة

### 1. Box Blur (`--filter box`)
- أبسط نوع blur
- كل الـ pixels في الـ kernel بيبقى وزنها 1 (متوسط بسيط)
- نتيجته مش smooth زي الـ Gaussian

### 2. Gaussian Blur (`--filter gaussian`)
- الـ blur الأكتر استخداماً
- الـ pixels القريبة من المركز بيبقى وزنها أكبر
- نتيجته أكثر طبيعية

### 3. Sharpen (`--filter sharpen`)
- بيزود وضوح الصورة
- بيستخدم Unsharp Masking: `output = input + 1.5*(input - blur)`

### 4. Sobel Edge Detection (`--filter sobel`)
- بيكتشف الحواف في الصورة
- بيستخدم kernel-ين: Gx (أفقي) وGy (رأسي)
- النتيجة = sqrt(Gx² + Gy²)

---

## 🧠 الفكرة التقنية

### Serial
- Loop عادي على كل pixel
- كل pixel بيحسب الـ convolution مع الـ kernel

### OpenMP
- `#pragma omp parallel for` على الـ outer loop (rows)
- كل thread بياخد مجموعة rows يشتغل عليها

### CUDA - Shared Memory Tiling
```
الصورة الكاملة (في GPU Global Memory)
         ↓
    تقسيم لـ Tiles
         ↓
كل Block بيحمّل Tile + Halo في Shared Memory
         ↓
كل Thread بيحسب Pixel واحد من الـ Shared Memory
         ↓
النتيجة ترجع للـ Global Memory
```

الـ Halo هي الـ pixels الإضافية حوالين الـ tile اللي محتاجها الـ threads على الحافة.

---

## 🔍 التحقق من الصحة

```bash
# قارن نتيجة الـ serial والـ OMP بصرياً
./build/image_filter --image test.jpg --filter gaussian --kernel 7 --impl serial
./build/image_filter --image test.jpg --filter gaussian --kernel 7 --impl omp --threads 4
# افتح الملفين في أي image viewer وقارنهم - المفروض يكونوا متطابقين

# أو من command line:
diff <(xxd results/gaussian_serial_k7.png) <(xxd results/gaussian_omp_k7.png)
```

---

## 📈 النتائج المتوقعة

| Implementation | Speedup المتوقع (HD Image) |
|---------------|---------------------------|
| Serial (baseline) | 1x |
| OpenMP 8 threads | ~4-6x |
| CUDA | ~20-50x |

---
