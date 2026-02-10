#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ITERS=30
WARMUP=5
PARSERS=(jbin simdjson yyjson rapidjson cjson sajson)
FILES=(data/canada.json data/twitter.json data/citm_catalog.json
       data/integers.json data/string_array.json data/escape_heavy.json
       data/mesh.json data/mixed_types.json)
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

while [[ $# -gt 0 ]]; do
    case $1 in
        -n*) ITERS="${1#-n}"; shift ;;
        -w*) WARMUP="${1#-w}"; shift ;;
        -f)  shift; FILES=(); while [[ $# -gt 0 && $1 != -* ]]; do FILES+=("$1"); shift; done ;;
        -h|--help)
            echo "Usage: $0 [-n<iters>] [-w<warmup>] [-f file1.json file2.json ...]"
            exit 0 ;;
        *)   echo "Unknown option: $1"; exit 1 ;;
    esac
done

BOLD=$'\033[1m'
DIM=$'\033[2m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
RED=$'\033[31m'
RESET=$'\033[0m'

info()  { printf "%s==>%s %s%s%s\n" "$CYAN" "$RESET" "$BOLD" "$1" "$RESET"; }
ok()    { printf "%s  ok%s %s\n" "$GREEN" "$RESET" "$1"; }
warn()  { printf "%s  !!%s %s\n" "$YELLOW" "$RESET" "$1"; }
fail()  { printf "%s  !!%s %s\n" "$RED" "$RESET" "$1"; }

# ─── Download dependencies ──────────────────────────────────────────────────
download_deps() {
    info "Checking dependencies"

    if [ ! -f simdjson/simdjson.h ]; then
        mkdir -p simdjson
        curl -sL "https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.h" \
            -o simdjson/simdjson.h
        curl -sL "https://raw.githubusercontent.com/simdjson/simdjson/master/singleheader/simdjson.cpp" \
            -o simdjson/simdjson.cpp
        ok "simdjson (downloaded)"
    else
        ok "simdjson (cached)"
    fi

    if [ ! -f yyjson/yyjson.h ]; then
        mkdir -p yyjson
        curl -sL "https://raw.githubusercontent.com/ibireme/yyjson/master/src/yyjson.h" \
            -o yyjson/yyjson.h
        curl -sL "https://raw.githubusercontent.com/ibireme/yyjson/master/src/yyjson.c" \
            -o yyjson/yyjson.c
        ok "yyjson (downloaded)"
    else
        ok "yyjson (cached)"
    fi

    if [ ! -d rapidjson/include/rapidjson ]; then
        mkdir -p "$TMPDIR/rj"
        git clone --depth 1 https://github.com/Tencent/rapidjson.git "$TMPDIR/rj" 2>/dev/null
        mkdir -p rapidjson/include
        cp -r "$TMPDIR/rj/include/rapidjson" rapidjson/include/
        rm -rf "$TMPDIR/rj"
        ok "rapidjson (downloaded)"
    else
        ok "rapidjson (cached)"
    fi

    if [ ! -f cjson/cJSON.h ]; then
        mkdir -p cjson
        curl -sL "https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h" \
            -o cjson/cJSON.h
        curl -sL "https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c" \
            -o cjson/cJSON.c
        ok "cJSON (downloaded)"
    else
        ok "cJSON (cached)"
    fi

    if [ ! -f sajson/sajson.h ]; then
        mkdir -p sajson
        curl -sL "https://raw.githubusercontent.com/chadaustin/sajson/master/include/sajson.h" \
            -o sajson/sajson.h
        ok "sajson (downloaded)"
    else
        ok "sajson (cached)"
    fi
}

# ─── Build ───────────────────────────────────────────────────────────────────
build_all() {
    info "Building benchmarks"

    CC="${CC:-cc}"
    CXX="${CXX:-c++}"

    # jbin — try PGO first, fallback to plain
    if make bench-pgo >/dev/null 2>&1; then
        ok "jbin (PGO+LTO)"
    elif make bench >/dev/null 2>&1; then
        ok "jbin"
    else
        fail "jbin FAILED"; PARSERS=("${PARSERS[@]/jbin/}")
    fi

    # simdjson
    if make bench-simdjson >/dev/null 2>&1; then
        ok "simdjson"
    else
        fail "simdjson FAILED"; PARSERS=("${PARSERS[@]/simdjson/}")
    fi

    # yyjson
    if $CC -O3 -march=native -std=c11 -fno-stack-protector \
        -D_POSIX_C_SOURCE=199309L \
        -o bench-yyjson bench_yyjson.c yyjson/yyjson.c -lm 2>/dev/null; then
        ok "yyjson"
    else
        fail "yyjson FAILED"; PARSERS=("${PARSERS[@]/yyjson/}")
    fi

    # rapidjson
    if $CXX -O3 -march=native -Irapidjson/include \
        -o bench-rapidjson bench_rapidjson.cpp 2>/dev/null; then
        ok "rapidjson"
    else
        fail "rapidjson FAILED"; PARSERS=("${PARSERS[@]/rapidjson/}")
    fi

    # cJSON
    if $CC -O3 -march=native -std=c11 -fno-stack-protector \
        -D_POSIX_C_SOURCE=199309L \
        -o bench-cjson bench_cjson.c cjson/cJSON.c -lm 2>/dev/null; then
        ok "cJSON"
    else
        fail "cJSON FAILED"; PARSERS=("${PARSERS[@]/cjson/}")
    fi

    # sajson
    if $CXX -O3 -march=native \
        -o bench-sajson bench_sajson.cpp 2>/dev/null; then
        ok "sajson"
    else
        fail "sajson FAILED"; PARSERS=("${PARSERS[@]/sajson/}")
    fi
}

# ─── Run ─────────────────────────────────────────────────────────────────────
run_all() {
    info "Running benchmarks (${ITERS} iterations, ${WARMUP} warmup)"
    echo

    for parser in "${PARSERS[@]}"; do
        [[ -z "$parser" ]] && continue
        local bin="./bench-${parser}"
        [[ "$parser" == "jbin" ]] && bin="./bench"

        if [ ! -x "$bin" ]; then
            warn "Skipping $parser (not built)"
            continue
        fi

        if ! $bin -n"${ITERS}" -w"${WARMUP}" "${FILES[@]}" \
             > "$TMPDIR/${parser}.txt" 2>/dev/null; then
            warn "$parser: some files failed"
        fi
        cat "$TMPDIR/${parser}.txt"
        echo
    done
}

# ─── Comparison table ────────────────────────────────────────────────────────
comparison_table() {
    # Filter to only parsers that produced output
    local active=()
    for p in "${PARSERS[@]}"; do
        [[ -z "$p" ]] && continue
        [[ -f "$TMPDIR/${p}.txt" ]] && active+=("$p")
    done

    if [[ ${#active[@]} -lt 2 ]]; then
        warn "Need at least 2 parsers for comparison"
        return
    fi

    info "GB/s comparison (higher is better)"
    echo

    # Header
    printf "%s%-25s" "$BOLD" "file"
    for p in "${active[@]}"; do
        printf " %10s" "$p"
    done
    printf "%s\n" "$RESET"

    # Separator
    printf "%-25s" "-------------------------"
    for _ in "${active[@]}"; do
        printf " ----------"
    done
    printf "\n"

    # Rows
    for file in "${FILES[@]}"; do
        local vals=()
        local max_val="0"

        # Collect GB/s for each parser
        for p in "${active[@]}"; do
            local v
            v=$(awk -v f="$file" '$1 == f {print $NF}' "$TMPDIR/${p}.txt" 2>/dev/null || echo "")
            vals+=("${v:---}")
            if [[ "$v" =~ ^[0-9] ]]; then
                local gt
                gt=$(awk "BEGIN{print ($v > $max_val) ? 1 : 0}")
                [[ "$gt" == "1" ]] && max_val="$v"
            fi
        done

        # Print with winner highlighted
        printf "%-25s" "$file"
        for v in "${vals[@]}"; do
            if [[ "$v" =~ ^[0-9] ]] && [[ $(awk "BEGIN{print ($v >= $max_val) ? 1 : 0}") == "1" ]]; then
                printf " %s%s%10s%s" "$GREEN" "$BOLD" "$v" "$RESET"
            else
                printf " %10s" "$v"
            fi
        done
        printf "\n"
    done

    echo
    printf "%s(green = fastest for that file)%s\n" "$DIM" "$RESET"

    # ─── Speedup vs jbin ─────────────────────────────────────────────
    if [[ -f "$TMPDIR/jbin.txt" ]]; then
        echo
        info "Speedup vs jbin (>1.00 = jbin is faster)"
        echo

        printf "%s%-25s" "$BOLD" "file"
        for p in "${active[@]}"; do
            [[ "$p" == "jbin" ]] && continue
            printf " %10s" "$p"
        done
        printf "%s\n" "$RESET"

        printf "%-25s" "-------------------------"
        for p in "${active[@]}"; do
            [[ "$p" == "jbin" ]] && continue
            printf " ----------"
        done
        printf "\n"

        for file in "${FILES[@]}"; do
            local jbin_gbs
            jbin_gbs=$(awk -v f="$file" '$1 == f {print $NF}' "$TMPDIR/jbin.txt" 2>/dev/null || echo "")
            [[ -z "$jbin_gbs" || "$jbin_gbs" == "--" ]] && continue

            printf "%-25s" "$file"
            for p in "${active[@]}"; do
                [[ "$p" == "jbin" ]] && continue
                local v
                v=$(awk -v f="$file" '$1 == f {print $NF}' "$TMPDIR/${p}.txt" 2>/dev/null || echo "")
                if [[ "$v" =~ ^[0-9] ]]; then
                    local ratio
                    ratio=$(awk "BEGIN{printf \"%.2f\", $jbin_gbs / $v}")
                    local faster
                    faster=$(awk "BEGIN{print ($jbin_gbs > $v) ? 1 : 0}")
                    if [[ "$faster" == "1" ]]; then
                        printf " %s%10sx%s" "$GREEN" "$ratio" "$RESET"
                    else
                        printf " %s%10sx%s" "$RED" "$ratio" "$RESET"
                    fi
                else
                    printf " %10s" "--"
                fi
            done
            printf "\n"
        done
        echo
        printf "%s(green = jbin wins, red = jbin loses)%s\n" "$DIM" "$RESET"
    fi
}

# ─── Main ────────────────────────────────────────────────────────────────────
download_deps
echo
build_all
echo
run_all
comparison_table
