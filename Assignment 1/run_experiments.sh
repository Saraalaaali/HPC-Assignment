#!/bin/bash
set -e

IMAGE=""
REPEATS=5
OUTPUT_DIR="results"
CSV_FILE="$OUTPUT_DIR/results.csv"
BINARY="./build/image_filter"

while [[ $# -gt 0 ]]; do
    case $1 in
        --image)   IMAGE="$2";   shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --output)  OUTPUT_DIR="$2"; shift 2 ;;
        --help) echo "Usage: $0 --image <path>"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$IMAGE" ];    then echo "❌ missing --image"; exit 1; fi
if [ ! -f "$IMAGE" ];  then echo "❌ image not found: $IMAGE"; exit 1; fi
if [ ! -f "$BINARY" ]; then echo "❌ binary not found, run: mkdir build && cd build && cmake .. && make -j\$(nproc) && cd .."; exit 1; fi

mkdir -p "$OUTPUT_DIR"
rm -f "$CSV_FILE"

echo "================================================"
echo "Image: $IMAGE   Repeats: $REPEATS"
echo "================================================"

run_one() {
    local filter="$1"
    local impl="$2"
    local kernel="$3"
    shift 3
    echo -n "  ▶ $filter | $impl | k=$kernel $@ ... "
    $BINARY \
        --image "$IMAGE" \
        --filter "$filter" \
        --impl "$impl" \
        --kernel "$kernel" \
        --repeats "$REPEATS" \
        --output "$OUTPUT_DIR" \
        --csv "$CSV_FILE" \
        "$@" 2>&1 | grep -E "(Time|Error)" || true
    echo "✅"
}

echo ""
echo "📊 Experiment 1: Serial Baseline"
for filter in box gaussian sharpen sobel; do
    for k in 3 7 11; do
        run_one "$filter" "serial" "$k"
    done
done

echo ""
echo "📊 Experiment 2: OpenMP Threads Scaling"
for filter in gaussian sobel; do
    for k in 3 7 11; do
        for threads in 1 2 4 8 16; do
            run_one "$filter" "omp" "$k" --threads "$threads"
        done
    done
done

echo ""
echo "📊 Experiment 3: CUDA Block Sizes"
for filter in gaussian sobel; do
    for k in 3 7 11; do
        for bx_by in "16 16" "32 8" "32 32"; do
            bx=$(echo $bx_by | cut -d' ' -f1)
            by=$(echo $bx_by | cut -d' ' -f2)
            run_one "$filter" "cuda" "$k" --block-x "$bx" --block-y "$by"
        done
    done
done

echo ""
echo "📈 Calculating speedup..."
python3 - << 'PYTHON_SCRIPT'
import csv, sys

with open("results/results.csv") as f:
    rows = list(csv.DictReader(f))

serial_times = {}
for r in rows:
    if r['impl'] == 'serial':
        serial_times[(r['filter'], r['kernel'])] = float(r['time_ms'])

for r in rows:
    st = serial_times.get((r['filter'], r['kernel']), 1.0)
    r['speedup'] = f"{st / float(r['time_ms']):.2f}"

fieldnames = list(rows[0].keys())
with open("results/results.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"{'Filter':<12} {'Impl':<8} {'Kernel':<8} {'Threads':<9} {'Time(ms)':<12} {'Speedup'}")
print("-" * 60)
for r in rows:
    if r['filter'] in ('gaussian','sobel') and r['kernel'] == '7':
        print(f"{r['filter']:<12} {r['impl']:<8} {r['kernel']:<8} {r['threads']:<9} {float(r['time_ms']):<12.2f} {r['speedup']}x")
PYTHON_SCRIPT

echo ""
echo "📊 Generating plots..."
python3 - << 'PLOT_SCRIPT'
import csv, sys
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("⚠️  run: pip3 install matplotlib --break-system-packages")
    sys.exit(0)

with open("results/results.csv") as f:
    rows = list(csv.DictReader(f))

# Plot 1: Serial Time vs Kernel Size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, fn in enumerate(['gaussian', 'sobel']):
    ax = axes[i]
    data = [r for r in rows if r['filter'] == fn and r['impl'] == 'serial']
    kernels = sorted(set(int(r['kernel']) for r in data))
    times   = [next(float(r['time_ms']) for r in data if int(r['kernel']) == k) for k in kernels]
    ax.bar([f"{k}x{k}" for k in kernels], times, color='steelblue')
    ax.set_title(f'{fn} — Serial Time vs Kernel Size')
    ax.set_xlabel('Kernel Size')
    ax.set_ylabel('Time (ms)')
    ax.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('results/plot_kernel_size.png', dpi=150)
print("✅ results/plot_kernel_size.png")

# Plot 2: OpenMP Speedup vs Threads
plt.figure(figsize=(10, 6))
for fn in ['gaussian', 'sobel']:
    data = [r for r in rows if r['filter']==fn and r['impl']=='omp' and int(r['kernel'])==7]
    if not data:
        continue
    data.sort(key=lambda r: int(r['threads']))
    plt.plot([int(r['threads']) for r in data],
             [float(r['speedup']) for r in data],
             marker='o', label=f'{fn} k=7')
plt.xlabel('Number of Threads')
plt.ylabel('Speedup vs Serial')
plt.title('OpenMP Speedup vs Threads')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('results/plot_omp_speedup.png', dpi=150)
print("✅ results/plot_omp_speedup.png")

# Plot 3: CUDA Speedup per Filter
plt.figure(figsize=(10, 6))
fnames = ['box', 'gaussian', 'sharpen', 'sobel']
speedups = []
for fn in fnames:
    r = next((r for r in rows if r['filter']==fn and r['impl']=='cuda' and int(r['kernel'])==7), None)
    speedups.append(float(r['speedup']) if r else 0)
plt.bar(fnames, speedups, color=['#e74c3c','#3498db','#2ecc71','#f39c12'])
plt.xlabel('Filter')
plt.ylabel('Speedup vs Serial')
plt.title('CUDA Speedup vs Serial (Kernel=7)')
plt.grid(axis='y', alpha=0.5)
plt.savefig('results/plot_cuda_speedup.png', dpi=150)
print("✅ results/plot_cuda_speedup.png")
PLOT_SCRIPT

echo ""
echo "================================================"
echo "✅ Done! Results in: $OUTPUT_DIR/"
echo "================================================"
