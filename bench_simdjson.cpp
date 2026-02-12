#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "simdjson/simdjson.h"

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
    simdjson::padded_string json;
    auto err = simdjson::padded_string::load(path).get(json);
    if (err) {
        printf("%-25s LOAD ERROR\n", path);
        return;
    }

    simdjson::dom::parser parser;
    auto doc = parser.parse(json);
    if (doc.error()) {
        printf("%-25s PARSE ERROR: %s\n", path, simdjson::error_message(doc.error()));
        return;
    }

    for (int i = 0; i < warmup; i++) {
        auto d = parser.parse(json);
        (void)d;
    }

    double *samples = (double *)malloc((size_t)iters * sizeof(double));
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        auto d = parser.parse(json);
        (void)d;
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

    double sz = (double)json.size();
    double mb = sz / (1024.0 * 1024.0);
    double gb = sz / (1024.0 * 1024.0 * 1024.0);
    printf("%-25s %7.2f MB  %8.3f  %8.3f  %8.3f  %7.3f  %8.3f\n",
           path, mb,
           best * 1000.0, median * 1000.0, worst * 1000.0,
           stddev * 1000.0, gb / median);

    free(samples);
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

    auto haswell = simdjson::get_available_implementations()["haswell"];
    if (haswell && haswell->supported_by_runtime_system())
        simdjson::get_active_implementation() = haswell;

    printf("simdjson DOM [%s] benchmark (%d iterations, %d warmup)\n",
           simdjson::get_active_implementation()->name().data(), iters, warmup);
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
        bench_file("data/apache_builds.json", iters, warmup);
        bench_file("data/booleans.json", iters, warmup);
        bench_file("data/deep_nested.json", iters, warmup);
        bench_file("data/escape_heavy.json", iters, warmup);
        bench_file("data/flat_kv.json", iters, warmup);
        bench_file("data/github_events.json", iters, warmup);
        bench_file("data/instruments.json", iters, warmup);
        bench_file("data/integers.json", iters, warmup);
        bench_file("data/mesh.json", iters, warmup);
        bench_file("data/mesh.pretty.json", iters, warmup);
        bench_file("data/mixed_types.json", iters, warmup);
        bench_file("data/string_array.json", iters, warmup);
        bench_file("data/truenull.json", iters, warmup);
        bench_file("data/update-center.json", iters, warmup);
        bench_file("data/whitespace.json", iters, warmup);
    }

    return 0;
}
