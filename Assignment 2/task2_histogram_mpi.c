/*
 * task2_histogram_mpi.c
 * =====================
 * Task 2: Histogram of Images — MPI Scatter/Gather
 * Compiled with: mpicxx task2_histogram_mpi.c -o task2_histogram_mpi
 *                $(pkg-config --cflags --libs opencv4) -lm
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

#define MASTER_RANK    0
#define MAX_PATH_LEN   512
#define MAX_IMAGES     4096
#define NUM_CHANNELS   3
#define MAX_PIXEL_VAL  256
#define TAG_COUNT      10
#define TAG_PATHS      11
#define TAG_RESULTS    12
#define TAG_HISTS      13

/* ── is supported image extension ── */
static int is_image(const char *name) {
    const char *e = strrchr(name, '.');
    if (!e) return 0;
    return strcasecmp(e,".jpg")==0 || strcasecmp(e,".jpeg")==0 ||
           strcasecmp(e,".png")==0 || strcasecmp(e,".bmp")==0  ||
           strcasecmp(e,".ppm")==0 || strcasecmp(e,".tiff")==0 ||
           strcasecmp(e,".tif")==0;
}

/* ── collect all image paths in a directory ── */
static int collect_images(const char *dir, char paths[][MAX_PATH_LEN], int max) {
    DIR *d = opendir(dir);
    if (!d) { fprintf(stderr,"[Master] Cannot open dir '%s'\n",dir); return 0; }
    int n = 0;
    struct dirent *e;
    while ((e = readdir(d)) && n < max)
        if (is_image(e->d_name))
            snprintf(paths[n++], MAX_PATH_LEN, "%s/%s", dir, e->d_name);
    closedir(d);
    return n;
}

/* ── compute BGR histogram for one image ── */
static int compute_histogram(const char *path, int window_size,
                              float *out, int *num_bins_out) {
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr,"[Worker] Cannot load '%s'\n", path);
        return 0;
    }
    int bins = MAX_PIXEL_VAL / window_size;
    if (bins < 1) bins = 1;
    *num_bins_out = bins;

    Mat planes[3];
    split(img, planes);
    float range[]         = {0.0f, (float)MAX_PIXEL_VAL};
    const float *ranges[] = {range};
    int hist_size[]       = {bins};

    for (int c = 0; c < NUM_CHANNELS; c++) {
        Mat hist;
        calcHist(&planes[c], 1, 0, Mat(), hist, 1, hist_size, ranges, true, false);
        for (int b = 0; b < bins; b++)
            out[c * bins + b] = hist.at<float>(b);
    }
    return 1;
}

/* ── write JSON output ── */
static void save_json(const char *path, int window_size, int total,
                      char names[][MAX_PATH_LEN], int *ok, int *workers,
                      float *hists, int bins, int n)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr,"[Master] Cannot write '%s'\n", path); return; }
    const char *ch[] = {"B","G","R"};
    int stride = NUM_CHANNELS * bins;
    fprintf(fp, "{\n  \"window_size\": %d,\n  \"total_images\": %d,\n"
                "  \"total_results\": %d,\n  \"results\": [\n", window_size, total, n);
    for (int i = 0; i < n; i++) {
        fprintf(fp, "    { \"image\": \"%s\", \"status\": \"%s\", \"worker\": %d",
                names[i], ok[i]?"ok":"error", workers[i]);
        if (ok[i]) {
            fprintf(fp, ",\n      \"histograms\": {");
            for (int c = 0; c < NUM_CHANNELS; c++) {
                fprintf(fp, "\n        \"%s\": [", ch[c]);
                for (int b = 0; b < bins; b++)
                    fprintf(fp, "%.0f%s", hists[i*stride+c*bins+b], b<bins-1?", ":"");
                fprintf(fp, "]%s", c<NUM_CHANNELS-1?",":"");
            }
            fprintf(fp, "\n      }");
        }
        fprintf(fp, "\n    }%s\n", i<n-1?",":"");
    }
    fprintf(fp, "  ]\n}\n");
    fclose(fp);
}

/* ════════════════════════════════════════
   MASTER
   ════════════════════════════════════════ */
static void master(int size, const char *dir, int wsize, const char *out_path) {
    printf("[Master] Started — %d processes total\n", size); fflush(stdout);

    static char all[MAX_IMAGES][MAX_PATH_LEN];
    int total = collect_images(dir, all, MAX_IMAGES);
    if (total == 0) {
        fprintf(stderr,"[Master] No images found in '%s'\n", dir);
        for (int w=1;w<size;w++) { int z=0; MPI_Send(&z,1,MPI_INT,w,TAG_COUNT,MPI_COMM_WORLD); }
        return;
    }
    printf("[Master] Found %d image(s), distributing to %d worker(s)\n",
           total, size-1); fflush(stdout);

    /* SCATTER — balanced chunks */
    int nw = size - 1, base = total/nw, extra = total%nw, start = 0;
    for (int w = 0; w < nw; w++) {
        int dst = w+1;
        int chunk = base + (w < extra ? 1 : 0);
        MPI_Send(&chunk, 1, MPI_INT, dst, TAG_COUNT, MPI_COMM_WORLD);
        MPI_Send(all[start], chunk*MAX_PATH_LEN, MPI_CHAR, dst, TAG_PATHS, MPI_COMM_WORLD);
        printf("[Master] → Worker %d: %d image(s)\n", dst, chunk); fflush(stdout);
        start += chunk;
    }
    printf("[Master] All chunks sent, waiting for results...\n"); fflush(stdout);

    /* GATHER */
    int bins = MAX_PIXEL_VAL/wsize; if(bins<1)bins=1;
    int stride = NUM_CHANNELS * bins;
    static char  rnames[MAX_IMAGES][MAX_PATH_LEN];
    static int   rok[MAX_IMAGES], rwk[MAX_IMAGES], meta[MAX_IMAGES*2];
    float *rhists = (float*)calloc(total * stride, sizeof(float));

    int got = 0;
    for (int w = 0; w < nw; w++) {
        int src = w+1, n = 0;
        MPI_Recv(&n, 1, MPI_INT, src, TAG_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (n <= 0) { printf("[Master] WARNING: Worker %d sent 0 results\n",src); continue; }
        MPI_Recv(rnames[got], n*MAX_PATH_LEN, MPI_CHAR, src, TAG_PATHS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(meta, n*2, MPI_INT, src, TAG_RESULTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i=0;i<n;i++) { rok[got+i]=meta[i*2]; rwk[got+i]=meta[i*2+1]; }
        MPI_Recv(&rhists[got*stride], n*stride, MPI_FLOAT, src, TAG_HISTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Master] ✓ Received %d result(s) from Worker %d\n", n, src); fflush(stdout);
        got += n;
    }

    save_json(out_path, wsize, total, rnames, rok, rwk, rhists, bins, got);
    int ok=0; for(int i=0;i<got;i++) if(rok[i]) ok++;
    printf("\n[Master] Done! %d OK, %d failed — saved to %s\n", ok, got-ok, out_path);
    free(rhists);
}

/* ════════════════════════════════════════
   WORKER
   ════════════════════════════════════════ */
static void worker(int rank, int wsize) {
    printf("[Worker %d] Ready\n", rank); fflush(stdout);

    int chunk = 0;
    MPI_Recv(&chunk, 1, MPI_INT, MASTER_RANK, TAG_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (chunk == 0) {
        printf("[Worker %d] Empty chunk, done\n", rank);
        MPI_Send(&chunk, 1, MPI_INT, MASTER_RANK, TAG_COUNT, MPI_COMM_WORLD);
        return;
    }

    static char paths[MAX_IMAGES][MAX_PATH_LEN];
    MPI_Recv(paths, chunk*MAX_PATH_LEN, MPI_CHAR, MASTER_RANK, TAG_PATHS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("[Worker %d] Processing %d image(s)\n", rank, chunk); fflush(stdout);

    int bins = MAX_PIXEL_VAL/wsize; if(bins<1)bins=1;
    int stride = NUM_CHANNELS * bins;
    static char rnames[MAX_IMAGES][MAX_PATH_LEN];
    static int  meta[MAX_IMAGES*2];
    float *hbuf = (float*)calloc(chunk * stride, sizeof(float));

    for (int i = 0; i < chunk; i++) {
        const char *sl = strrchr(paths[i], '/');
        strncpy(rnames[i], sl ? sl+1 : paths[i], MAX_PATH_LEN-1);
        int nb = bins;
        int ok = compute_histogram(paths[i], wsize, &hbuf[i*stride], &nb);
        meta[i*2]=ok; meta[i*2+1]=rank;
        printf("[Worker %d] %s %s\n", rank, ok?"OK  ":"FAIL", rnames[i]); fflush(stdout);
    }

    MPI_Send(&chunk,         1,              MPI_INT,   MASTER_RANK, TAG_COUNT,   MPI_COMM_WORLD);
    MPI_Send(rnames,         chunk*MAX_PATH_LEN, MPI_CHAR,  MASTER_RANK, TAG_PATHS,   MPI_COMM_WORLD);
    MPI_Send(meta,           chunk*2,        MPI_INT,   MASTER_RANK, TAG_RESULTS, MPI_COMM_WORLD);
    MPI_Send(hbuf,           chunk*stride,   MPI_FLOAT, MASTER_RANK, TAG_HISTS,   MPI_COMM_WORLD);
    printf("[Worker %d] Sent %d result(s) to master\n", rank, chunk); fflush(stdout);
    free(hbuf);
}

/* ════════════════════════════════════════
   MAIN
   ════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank==0) fprintf(stderr,"Need at least 2 processes\n");
        MPI_Finalize(); return 1;
    }
    if (rank == MASTER_RANK) {
        if (argc < 4) {
            fprintf(stderr,"Usage: mpirun -np <N> ./task2_histogram_mpi "
                           "<images_dir> <window_size> <output.json>\n");
            MPI_Abort(MPI_COMM_WORLD,1); return 1;
        }
        master(size, argv[1], atoi(argv[2]), argv[3]);
    } else {
        worker(rank, argc>=3 ? atoi(argv[2]) : 8);
    }
    MPI_Finalize();
    return 0;
}
