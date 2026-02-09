#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "simdjson/simdjson.h"

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void bench_file(const char *path, int iters) {
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

    double best = 1e30;
    for (int i = 0; i < iters; i++) {
        double t0 = now_sec();
        auto d = parser.parse(json);
        (void)d;
        double t1 = now_sec();
        double elapsed = t1 - t0;
        if (elapsed < best) best = elapsed;
    }

    double sz = (double)json.size();
    double mb = sz / (1024.0 * 1024.0);
    double gb = sz / (1024.0 * 1024.0 * 1024.0);
    printf("%-25s %8.2f MB  %8.3f ms  %8.3f GB/s\n",
           path, mb, best * 1000.0, gb / best);
}

int main(int argc, char **argv) {
    int iters = 10;
    if (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'n') {
        iters = atoi(argv[1] + 2);
        argc--; argv++;
    }

    auto haswell = simdjson::get_available_implementations()["haswell"];
    if (haswell && haswell->supported_by_runtime_system())
        simdjson::get_active_implementation() = haswell;

    printf("simdjson DOM [%s] benchmark (%d iterations, best-of)\n",
           simdjson::get_active_implementation()->name().data(), iters);
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

    return 0;
}
