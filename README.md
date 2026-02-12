# jbin

A high-performance freestanding C11 JSON parser.

- **~2000 lines** of C11 across two files (`jbin.h` + `jbin.c`)
- **AVX2 SIMD** two-pass architecture for large inputs (structural index via PCLMULQDQ + pshufb classification)
- **Zero-copy strings** — unescaped strings reference the input buffer directly
- **Freestanding** — no libc dependency, uses a fixed-size arena allocator
- **Tree DOM** — produces a linked node tree (objects/arrays with first_child/next pointers)

## Benchmarks

AMD Ryzen 5 7500F (Zen 4), Clang PGO+LTO, median of 1000 iterations.

### All parsers (GB/s, standard benchmark files)

| File | Size | jbin | simdjson | yyjson | sajson | rapidjson | cJSON |
|---|---|---|---|---|---|---|---|
| canada.json | 2.15 MB | **2.19** | 1.46 | 1.34 | 0.62 | 0.72 | 0.13 |
| citm_catalog.json | 1.65 MB | **6.53** | 5.59 | 3.88 | 1.94 | 1.15 | 0.69 |
| twitter.json | 0.60 MB | 4.55 | **5.53** | 3.57 | 2.22 | 0.72 | 0.54 |

### jbin vs simdjson (GB/s, full suite)

| File | Size | jbin | simdjson | Winner |
|---|---|---|---|---|
| canada.json | 2.15 MB | **2.19** | 1.46 | jbin 1.49x |
| citm_catalog.json | 1.65 MB | **6.53** | 5.59 | jbin 1.17x |
| flat_kv.json | 5.57 MB | **6.97** | 6.46 | jbin 1.08x |
| mesh.pretty.json | 1.50 MB | **4.38** | 2.44 | jbin 1.80x |
| instruments.json | 0.21 MB | **5.48** | 4.55 | jbin 1.21x |
| integers.json | 5.24 MB | **1.12** | 0.88 | jbin 1.27x |
| deep_nested.json | 0.43 MB | **2.79** | 2.07 | jbin 1.35x |
| mixed_types.json | 3.96 MB | **1.93** | 1.67 | jbin 1.16x |
| apache_builds.json | 0.12 MB | **6.49** | 4.73 | jbin 1.37x |
| github_events.json | 0.06 MB | **6.98** | 6.02 | jbin 1.16x |
| booleans.json | 1.02 MB | **1.00** | 0.91 | jbin 1.10x |
| mesh.json | 0.69 MB | **2.36** | 1.46 | jbin 1.62x |
| truenull.json | 0.01 MB | **3.90** | 3.07 | jbin 1.27x |
| update-center.json | 0.51 MB | **5.53** | 4.43 | jbin 1.25x |
| whitespace.json | 0.01 MB | **34.63** | 19.96 | jbin 1.73x |
| twitter.json | 0.60 MB | 4.55 | **5.53** | simdjson 1.22x |
| escape_heavy.json | 5.18 MB | 1.18 | **2.58** | simdjson 2.18x |
| string_array.json | 6.53 MB | 1.21 | **1.85** | simdjson 1.53x |

jbin wins **15/18** files. It dominates on number-heavy, structural, and large inputs where SWAR/AVX2 digit scanning, SIMD structural indexing, and compact node layout give it an edge. simdjson wins on string-heavy inputs where its tape format avoids per-string node allocation.

## Building

```sh
# Run tests
make check

# Simple optimized benchmark build
make bench
./bench -n100

# PGO+LTO benchmark build (recommended for reproducing numbers above)
make bench-pgo-clang
./bench -n1000

# Verify freestanding compilation
make freestanding-check
```

## Reproducing Benchmarks

Comparison benchmarks require vendored library sources in the repo (`simdjson/`, `yyjson/`, `cjson/`, `rapidjson/`, `sajson/`).

```sh
# Build and run any comparison benchmark
make bench-simdjson  && ./bench-simdjson -n1000
make bench-yyjson    && ./bench-yyjson -n1000
make bench-rapidjson && ./bench-rapidjson -n1000
make bench-cjson     && ./bench-cjson -n1000
make bench-sajson    && ./bench-sajson -n1000
```

Some data files are generated. Recreate them before benchmarking:

```sh
python3 scripts/gen_large.py > data/large.json
```

## Usage

```c
#include "jbin.h"

// Configure arena sizes at compile time
// -DJBIN_MAX_NODES=4096
// -DJBIN_MAX_STRING=262144
// -DJBIN_MAX_DEPTH=256

JbinArena arena;
jbin_arena_init(&arena);

const char *json = "{\"name\":\"jbin\",\"fast\":true}";
JbinResult r = jbin_parse(&arena, json, strlen(json));

if (r.error != JBIN_OK) {
    printf("error: %s at position %u\n", jbin_error_str(r.error), r.error_pos);
    return 1;
}

// Walk the tree
JbinNode *root = &arena.nodes[r.root];           // JBIN_OBJECT
JbinNode *key  = &arena.nodes[root->first_child]; // "name"
JbinNode *val  = &arena.nodes[key->next];          // "jbin"

// Get string content (may reference input buffer or arena copy)
const char *str = jbin_str(&arena, val, json);
uint32_t len = val->str_len;
```

## Node Types

| Type | Fields |
|---|---|
| `JBIN_OBJECT` | `first_child` → first key node |
| `JBIN_ARRAY` | `first_child` → first element node |
| `JBIN_STRING` | `str_off`, `str_len` — use `jbin_str()` to get pointer |
| `JBIN_NUMBER` | `str_off`, `str_len` — stored as original text |
| `JBIN_TRUE` | — |
| `JBIN_FALSE` | — |
| `JBIN_NULL` | — |

All nodes have a `next` field (`JBIN_NONE` if last sibling). Object children alternate key/value pairs.

## Data Files

Standard benchmark files (`canada.json`, `twitter.json`, `citm_catalog.json`) are included. Large generated files can be recreated:

```sh
# Generate large.json (~119 MB, 500K records)
python3 scripts/gen_large.py > data/large.json

# Generate with custom record count
python3 scripts/gen_large.py 1000000 > data/large.json
```

## License

MIT
