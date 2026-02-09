# jbin

A high-performance freestanding C11 JSON parser.

- **~2000 lines** of C11 across two files (`jbin.h` + `jbin.c`)
- **AVX2 SIMD** two-pass architecture for large inputs (structural index via PCLMULQDQ + pshufb classification)
- **Zero-copy strings** — unescaped strings reference the input buffer directly
- **Freestanding** — no libc dependency, uses a fixed-size arena allocator
- **Tree DOM** — produces a linked node tree (objects/arrays with first_child/next pointers)

## Benchmarks

Head-to-head vs simdjson (DOM API) on AMD Ryzen 5 7500F (Zen 4), GCC 13.3, PGO+LTO, best-of-1000:

| File | Size | jbin (GB/s) | simdjson (GB/s) | Winner |
|---|---|---|---|---|
| canada.json | 2.15 MB | **2.48** | 1.37 | jbin 1.82x |
| citm_catalog.json | 1.65 MB | **5.71** | 4.09 | jbin 1.39x |
| large.json | 118.78 MB | **2.22** | 1.88 | jbin 1.19x |
| flat_kv.json | 5.57 MB | **6.21** | 3.92 | jbin 1.58x |
| mesh.pretty.json | 1.50 MB | **4.17** | 2.29 | jbin 1.82x |
| instruments.json | 0.21 MB | **5.02** | 3.77 | jbin 1.33x |
| integers.json | 5.24 MB | **1.17** | 0.91 | jbin 1.29x |
| deep_nested.json | 0.43 MB | **2.13** | 1.74 | jbin 1.23x |
| numbers.json | 12.67 MB | **1.60** | 1.38 | jbin 1.16x |
| mixed_types.json | 3.96 MB | **1.73** | 1.49 | jbin 1.17x |
| apache_builds.json | 0.12 MB | **5.50** | 4.93 | jbin 1.12x |
| github_events.json | 0.06 MB | **5.91** | 5.27 | jbin 1.12x |
| github_events_large.json | 24.93 MB | **3.57** | 3.51 | jbin 1.02x |
| mesh.json | 0.69 MB | 1.35 | **1.40** | simdjson 1.04x |
| update-center.json | 0.51 MB | 4.31 | **4.38** | simdjson 1.02x |
| twitter.json | 0.60 MB | 3.28 | **4.72** | simdjson 1.44x |
| escape_heavy.json | 5.18 MB | 1.20 | **2.00** | simdjson 1.67x |
| string_array.json | 6.53 MB | 1.23 | **2.06** | simdjson 1.67x |

jbin wins **14/18** categories. It dominates on number-heavy, structural, and large inputs where SWAR/AVX2 digit scanning, SIMD structural indexing, and compact node layout give it an edge. simdjson wins on string-heavy inputs where its tape format avoids per-string node allocation.

## Building

```sh
# Run tests
make check

# Simple optimized benchmark build
make bench
./bench -n100

# PGO+LTO benchmark build (recommended)
make bench-pgo
./bench -n1000

# Compare against simdjson
make bench-simdjson
./bench-simdjson -n1000

# Verify freestanding compilation
make freestanding-check
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
