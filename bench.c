#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "jbin.h"

static JbinArena *arena;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_file(const char *path, int iters) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); return; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz);
    fread(buf, 1, (size_t)sz, f);
    fclose(f);

    jbin_arena_init(arena);
    JbinResult r = jbin_parse(arena, buf, (uint32_t)sz);
    if (r.error != JBIN_OK) {
        printf("%-25s PARSE ERROR: %s @%u\n", path, jbin_error_str(r.error), r.error_pos);
        free(buf);
        return;
    }

    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        jbin_arena_init(arena);
        double t0 = now_sec();
        jbin_parse(arena, buf, (uint32_t)sz);
        double t1 = now_sec();
        double elapsed = t1 - t0;
        if (elapsed < best) best = elapsed;
    }

    double mb = (double)sz / (1024.0 * 1024.0);
    double gb = (double)sz / (1024.0 * 1024.0 * 1024.0);
    printf("%-25s %8.2f MB  %8.3f ms  %8.3f GB/s\n",
           path, mb, best * 1000.0, gb / best);

    free(buf);
}

int main(int argc, char **argv) {
    arena = malloc(sizeof(JbinArena));
    if (!arena) { printf("arena alloc failed\n"); return 1; }

    int iters = 10;
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'n') {
        iters = atoi(argv[1] + 2);
        argc--; argv++;
    }

    printf("jbin benchmark (%d iterations, best-of)\n", iters);
    printf("%-25s %8s  %10s  %10s\n", "file", "size", "time", "throughput");
    printf("--------------------------------------------------------------\n");

    if (argc > 1) {
        for (int i = 1; i < argc; i++)
            bench_file(argv[i], iters);
    } else {
        bench_file("data/canada.json", iters);
        bench_file("data/twitter.json", iters);
        bench_file("data/citm_catalog.json", iters);
        bench_file("data/large.json", iters);
    }

    free(arena);
    return 0;
}
