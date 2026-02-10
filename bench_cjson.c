#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cjson/cJSON.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static void bench_file(const char *path, int iters, int warmup) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("%-25s LOAD ERROR\n", path); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    fread(buf, 1, (size_t)sz, f);
    buf[sz] = '\0';
    fclose(f);

    cJSON *root = cJSON_ParseWithLength(buf, (size_t)sz);
    if (!root) {
        printf("%-25s PARSE ERROR\n", path);
        free(buf);
        return;
    }
    cJSON_Delete(root);

    for (int i = 0; i < warmup; i++) {
        root = cJSON_ParseWithLength(buf, (size_t)sz);
        cJSON_Delete(root);
    }

    double *samples = malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        root = cJSON_ParseWithLength(buf, (size_t)sz);
        cJSON_Delete(root);
        double t1 = now_sec();
        samples[i] = t1 - t0;
    }

    qsort(samples, (size_t)iters, sizeof(double), cmp_double);

    double best   = samples[0];
    double median  = samples[iters / 2];
    double worst   = samples[iters - 1];

    double sum = 0;
    for (int i = 0; i < iters; i++) sum += samples[i];
    double mean = sum / iters;
    double var = 0;
    for (int i = 0; i < iters; i++) {
        double d = samples[i] - mean;
        var += d * d;
    }
    double stddev = sqrt(var / iters);

    double mb = (double)sz / (1024.0 * 1024.0);
    double gb = (double)sz / (1024.0 * 1024.0 * 1024.0);
    printf("%-25s %7.2f MB  %8.3f  %8.3f  %8.3f  %7.3f  %8.3f\n",
           path, mb,
           best * 1000.0, median * 1000.0, worst * 1000.0,
           stddev * 1000.0, gb / median);

    free(samples);
    free(buf);
}

int main(int argc, char **argv) {
    int iters = 10;
    int warmup = 3;

    while (argc > 1 && argv[1][0] == '-') {
        if (argv[1][1] == 'n')
            iters = atoi(argv[1] + 2);
        else if (argv[1][1] == 'w')
            warmup = atoi(argv[1] + 2);
        argc--; argv++;
    }

    printf("cJSON benchmark (%d iterations, %d warmup)\n", iters, warmup);
    printf("%-25s %9s  %8s  %8s  %8s  %7s  %8s\n",
           "file", "size", "min", "median", "max", "stddev", "GB/s");
    printf("----------------------------"
           "------------------------------------------------------\n");

    if (argc > 1) {
        for (int i = 1; i < argc; i++)
            bench_file(argv[i], iters, warmup);
    } else {
        bench_file("data/canada.json", iters, warmup);
        bench_file("data/twitter.json", iters, warmup);
        bench_file("data/citm_catalog.json", iters, warmup);
        bench_file("data/large.json", iters, warmup);
    }

    return 0;
}
