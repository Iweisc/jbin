# jbin

A high-performance freestanding C11 JSON parser.

- **~1600 lines** of C11 across two files (`jbin.h` + `jbin.c`)
- **AVX2 SIMD** two-pass architecture for large inputs (structural index via PCLMULQDQ + pshufb classification)
- **Zero-copy strings** — unescaped strings reference the input buffer directly
- **Freestanding** — no libc dependency, uses a fixed-size arena allocator
- **Tree DOM** — produces a linked node tree (objects/arrays with first_child/next pointers)

## Benchmarks

Head-to-head vs simdjson (DOM API) on AMD Ryzen 5 7500F (Zen 4), GCC 13.3, PGO+LTO, pinned to single core, best-of-500:

| File | Size | jbin (GB/s) | simdjson (GB/s) | Winner |
|---|---|---|---|---|
| canada.json | 2.15 MB | **2.01** | 1.50 | jbin 1.34x |
| citm_catalog.json | 1.65 MB | **5.68** | 4.86 | jbin 1.17x |
| large.json | 118.78 MB | **2.12** | 1.07 | jbin 1.98x |
| integers.json | 5.24 MB | **1.12** | 0.54 | jbin 2.09x |
| mesh.pretty.json | 1.50 MB | **2.88** | 2.56 | jbin 1.13x |
| instruments.json | 0.21 MB | **4.89** | 4.78 | jbin 1.02x |
| mixed_types.json | 3.96 MB | 1.69 | **1.70** | ~tie |
| numbers.json | 12.67 MB | 0.89 | **1.01** | simdjson 1.14x |
| flat_kv.json | 5.57 MB | 6.18 | **7.34** | simdjson 1.19x |
| twitter.json | 0.60 MB | 3.11 | **5.59** | simdjson 1.80x |
| github_events_large.json | 24.93 MB | 2.07 | **4.34** | simdjson 2.10x |
| escape_heavy.json | 5.18 MB | 0.51 | **2.05** | simdjson 4.01x |
| string_array.json | 6.53 MB | 0.26 | **1.90** | simdjson 7.46x |

jbin wins on number-heavy and large inputs where SIMD structural indexing and compact node layout dominate. simdjson wins on string-heavy inputs where its lazy/zero-copy tape format avoids per-string decode work.

## Building

```sh
# Run tests
make check

# Simple optimized benchmark build
make bench
./bench -n100

# PGO+LTO benchmark build (recommended)
make bench-pgo
./bench -n500

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
