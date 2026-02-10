#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include "sajson/sajson.h"

static double now_sec() {
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
    char *buf = (char *)malloc((size_t)sz);
    fread(buf, 1, (size_t)sz, f);
    fclose(f);

    /* sajson mutates input — need a working copy */
    char *copy = (char *)malloc((size_t)sz);

    memcpy(copy, buf, (size_t)sz);
    sajson::document doc = sajson::parse(
        sajson::dynamic_allocation(),
        sajson::mutable_string_view((size_t)sz, copy));
    if (!doc.is_valid()) {
        printf("%-25s PARSE ERROR: %s at %zu\n", path,
               doc.get_error_message_as_cstring(), doc.get_error_line());
        free(copy); free(buf);
        return;
    }

    for (int i = 0; i < warmup; i++) {
        memcpy(copy, buf, (size_t)sz);
        sajson::document d = sajson::parse(
            sajson::dynamic_allocation(),
            sajson::mutable_string_view((size_t)sz, copy));
        (void)d;
    }

    double *samples = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        memcpy(copy, buf, (size_t)sz);
        double t0 = now_sec();
        sajson::document d = sajson::parse(
            sajson::dynamic_allocation(),
            sajson::mutable_string_view((size_t)sz, copy));
        double t1 = now_sec();
        (void)d;
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
    free(copy);
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

    printf("sajson benchmark (%d iterations, %d warmup)\n", iters, warmup);
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
