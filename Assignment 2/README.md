# Task 2 — MPI Color Histogram of Images

Distributed image histogram computation using **MPI** (C++) across a Docker-based multi-node cluster.

## How it works

1. The **master** process scans the images directory and distributes image paths to workers (scatter).
2. Each **worker** computes a 3-channel (BGR) color histogram for its assigned images using OpenCV.
3. Workers send results back to the master (gather).
4. The master writes a structured `histograms.json` output file.

## Project structure

```
.
├── Dockerfile                  # Single image for master + all workers
├── docker-compose.yml          # 1 master + 6 workers on two networks
├── hostfile                    # MPI host list (7 slots total)
├── run.sh                      # Cluster startup script
├── task2_histogram_mpi.c       # Main MPI + OpenCV C++ source
├── images/                     # Place input images here (jpg/png/bmp...)
└── output/                     # histograms.json is written here
```

## Quick start

```bash
# 1. Add images to the images/ directory (jpg, png, bmp, ppm, tiff)
mkdir -p images output
cp /path/to/your/images/*.jpg images/

# 2. Build and start the cluster
chmod +x run.sh
./run.sh        # choose 'b' to build on first run

# 3. Run with window_size=8
docker exec master mpirun \
  --allow-run-as-root \
  --hostfile /app/hostfile \
  -np 7 \
  --mca btl tcp,self \
  --mca btl_tcp_if_include 172.28.0.0/24,172.25.0.0/24 \
  /app/task2_histogram_mpi /app/images 8 /app/output/histograms.json
# 4. View the output
cat output/histograms.json
```

## Parameters

| Parameter      | Description                                      | Example |
|----------------|--------------------------------------------------|---------|
| `images_dir`   | Directory containing input images                | `/app/images` |
| `window_size`  | Bin width — how many pixel values per bin (1–256)| `8` → 32 bins, `16` → 16 bins |
| `output.json`  | Path for the JSON result file                    | `/app/output/histograms.json` |

**Bins formula:** `bins = 256 / window_size`

## Output format

```json
{
  "window_size": 8,
  "total_images": 10,
  "total_results": 10,
  "results": [
    {
      "image": "photo.jpg",
      "status": "ok",
      "worker": 2,
      "histograms": {
        "B": [120, 340, ...],
        "G": [98,  210, ...],
        "R": [300, 150, ...]
      }
    }
  ]
}
```

## Network topology

| Network            | Subnet          | Members                    |
|--------------------|-----------------|----------------------------|
| `mpi_network`      | 172.28.0.0/24   | master (`.100`), worker1-3 |
| `mpi_control_network` | 172.25.0.0/24 | master (`.100`), worker4-6 |

## Supported image formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`, `.tiff`, `.tif`
